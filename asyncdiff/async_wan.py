import torch.distributed as dist
import torch
from .tools import ResultPicker
from .pipe_config import splite_model


"""########################################

NOTE: This backend is currently INCOMPLETE.

########################################"""


class ModulePlugin(object):
    def __init__(self,module,  model_i, stride=1, run_mode=None, cached_step=1):
        self.model_i = model_i
        self.stride = stride
        self.run_mode = run_mode
        self.cached_step = cached_step
        self.module = module
        self.module.plugin = self
        self.init_state()
        self.inject_forward()
        self.rank = dist.get_rank()

    def init_state(self,warmup_n=1):
        self.warmup_n = warmup_n
        self.result_structure = None
        self.cached_result = None
        self.infer_step = 0

    def cache_sync(self, async_flag):
        if self.infer_step >= self.warmup_n:
            dist.broadcast(self.cached_result, self.model_i, async_op=async_flag)

    def inject_forward(self):
        assert not hasattr(self.module, 'old_forward'), "Module already has old_forward attribute."
        module = self.module
        module.old_forward = module.forward

        def new_forward(*args, **kwargs):
            run_locally = (self.run_mode[0] == self.model_i) and ((self.infer_step - 1) % self.stride == self.stride - 1) and (self.cached_step <= 1 or (self.infer_step - 1) % self.cached_step != 0)
            if self.infer_step < self.warmup_n:
                if self.rank == 0 or self.cached_result is None:
                    result = module.old_forward(*args, **kwargs)
                    c_r, r_s = ResultPicker.dump(result)
                    dist.broadcast(c_r, 0)
                    self.cached_result, self.result_structure = c_r, r_s
                else:
                    dist.broadcast(self.cached_result, 0)
                    result = ResultPicker.load(self.cached_result, self.result_structure)
            elif run_locally:
                result = module.old_forward(*args, **kwargs)
                # if (self.infer_step+1==self.warmup_n) or (self.infer_step + 1 > self.warmup_n and self.run_mode[1]==0):
                self.cached_result, self.result_structure = ResultPicker.dump(result)
            else:
                result = ResultPicker.load(self.cached_result, self.result_structure)
            self.infer_step += 1
            return result

        module.forward = new_forward

class AsyncDiff(object):
    def __init__(self, *args, **kwargs):
        # args
        self.pipeline = args[0].to(f"cuda:{dist.get_rank()}")
        torch.cuda.set_device(f"cuda:{dist.get_rank()}")
        self.pipe_id = args[1]

        # kwargs
        self.model_n = kwargs.get("model_n", 2)
        self.stride = kwargs.get("stride", 1)
        self.warm_up = kwargs.get("warm_up", 1)
        self.time_shift = kwargs.get("time_shift", 0)
        self.cached_step = kwargs.get("cached_step", 1)

        # other
        # dist.init_process_group("nccl")
        if not dist.get_rank():
            assert self.model_n + self.stride - 1 == dist.get_world_size(), "[ERROR]: The strategy is not compatible with the number of devices. (model_n + stride - 1) should be equal to world_size."
        self.reformed_modules = {}
        self.reform_pipeline()
        # step = 39 // self.model_n
        # self.comm_index = [(i + 1) * step for i in range(self.model_n - 1)]

    def reset_state(self,warm_up=1):
        self.warm_up = warm_up
        for each in self.reformed_modules.values():
            each.plugin.init_state(warmup_n=warm_up)

    def reform_module(self, module, module_id, model_i):
        run_mode = (dist.get_rank(), 0) if dist.get_rank() < self.model_n else (self.model_n - 1, 1)
        ModulePlugin(module, model_i, self.stride, run_mode, self.cached_step)
        self.reformed_modules[(model_i, module_id)] = module

    def reform_transformer(self):
        transformer = self.pipeline.transformer
        assert not hasattr(transformer, 'old_forward'), "transformer already has old_forward attribute."
        transformer.old_forward = transformer.forward

        def transformer_forward(*args, **kwargs):
            infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
            index = 1
            run_locally = (infer_step - 1) % self.stride == self.stride - 1 and (self.cached_step <= 1 or (infer_step - 1) % self.cached_step != 0)
            # if run_locally:
            for each in self.reformed_modules.values():
                # if index in self.comm_index:
                each.plugin.cache_sync(False)
                # index += 1

            # TODO: there are 2 steps for every 1 timestep
            """
            if infer_step >= self.warm_up and self.time_shift > 0:
                timestep = self.pipeline.scheduler.timesteps[infer_step-shift]
                timestep = timestep.expand(1)
                kwargs["timestep"] = timestep
            """

            sample = transformer.old_forward(*args, **kwargs)[0]
            infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
            if infer_step >= self.warm_up and run_locally:
                sample = sample.contiguous()
                dist.broadcast(sample, max(0, self.model_n - 1))
            return sample,

        transformer.forward = transformer_forward


    def reform_pipeline(self):
        models = splite_model(self.pipeline, self.pipe_id, self.model_n)
        for model_i, sub_model in enumerate(models):
            for module_id, module in enumerate(sub_model):
                self.reform_module(module, module_id, model_i)
        self.reform_transformer()
