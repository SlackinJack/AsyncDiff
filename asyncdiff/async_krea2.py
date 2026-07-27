import torch.distributed as dist
import torch
from .tools import ResultPicker
from .pipe_config import splite_model
from .utils import get_ramped_time_shift









class ModulePlugin(object):
    def __init__(self, module, model_i, stride=1, target_i=None, cached_step=1):
        self.model_i = model_i
        self.stride = stride
        self.target_i = target_i
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

    def cache_sync(self):
        if self.infer_step >= self.warmup_n:
            dist.broadcast(self.cached_result, self.model_i)

    def inject_forward(self):
        assert not hasattr(self.module, 'old_forward'), "Module already has old_forward attribute."
        module = self.module
        module.old_forward = module.forward

        def new_forward(*args, **kwargs):
            run_locally = (self.target_i == self.model_i) and ((self.infer_step - 1) % self.stride == self.stride - 1) and (self.cached_step <= 1 or (self.infer_step - 1) % self.cached_step != 0)
            if self.infer_step <= self.warmup_n:
                if self.rank == 0 or self.cached_result is None:
                    result = module.old_forward(*args, **kwargs)
                    self.cached_result, self.result_structure = ResultPicker.dump(result)
                    dist.broadcast(self.cached_result, 0)
                else:
                    dist.broadcast(self.cached_result, 0)
                    result = ResultPicker.load(self.cached_result, self.result_structure)
            elif run_locally:
                result = module.old_forward(*args, **kwargs)
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
        self.shifted_steps = kwargs.get("shifted_steps", 0)
        self.cached_step = kwargs.get("cached_step", 1)
        self.ramped_time_shift = kwargs.get("ramped_time_shift", False)

        # other
        # dist.init_process_group("nccl")
        if not dist.get_rank():
            assert self.model_n + self.stride - 1 == dist.get_world_size(), "[ERROR]: The strategy is not compatible with the number of devices. (model_n + stride - 1) should be equal to world_size."
        self.reformed_modules = {}
        self.reform_pipeline()
        # step = 28 // self.model_n
        # self.comm_index = [(i + 1) * step for i in range(self.model_n - 1)]

    def reset_state(self,warm_up=1):
        self.warm_up = warm_up
        for each in self.reformed_modules.values():
            each.plugin.init_state(warmup_n=warm_up)

    def reform_module(self, module, module_id, model_i):
        target_i = dist.get_rank() if dist.get_rank() < self.model_n else self.model_n - 1
        ModulePlugin(module, model_i, self.stride, target_i, self.cached_step)
        self.reformed_modules[(model_i, module_id)] = module

    def reform_transformer(self):
        transformer = self.pipeline.transformer
        assert not hasattr(transformer, 'old_forward'), "transformer already has old_forward attribute."
        transformer.old_forward = transformer.forward

        def transformer_forward(*args, **kwargs):
            infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
            # index = 1
            run_locally = (infer_step - 1) % self.stride == self.stride - 1 and (self.cached_step <= 1 or (infer_step - 1) % self.cached_step != 0)
            for each in self.reformed_modules.values():
                # if index in self.comm_index:
                each.plugin.cache_sync()
                # index += 1

            if self.time_shift > 0 and self.shifted_steps > 0 and infer_step <= self.shifted_steps:
                if self.ramped_time_shift:
                    new_shift = get_ramped_time_shift(self.time_shift, self.shifted_steps, infer_step)
                else:
                    new_shift = self.time_shift
                shift = max(0, infer_step - new_shift)
                device = kwargs["timestep"].device
                dtype = kwargs["timestep"].dtype
                timesteps = self.pipeline.scheduler.timesteps
                timestep = timesteps[shift].item() / len(timesteps)
                kwargs["timestep"] = torch.tensor(timestep, device=device, dtype=dtype).unsqueeze(0)

            sample = transformer.old_forward(*args, **kwargs)[0]

            infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
            if infer_step >= self.warm_up:
                dist.broadcast(sample, max(0, self.model_n - 1))
            return sample,

        transformer.forward = transformer_forward


    def reform_pipeline(self):
        models = splite_model(self.pipeline, self.pipe_id, self.model_n)
        for model_i, sub_model in enumerate(models):
            for module_id, module in enumerate(sub_model):
                self.reform_module(module, module_id, model_i)
        self.reform_transformer()
