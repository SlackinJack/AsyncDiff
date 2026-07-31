# AsyncDiff

For more information, please visit the [original repository](https://github.com/czg1225/AsyncDiff).


## Breaking Changes:
- Manual init of distributed environment
- Requires an additional `pipeline_type` argument (to set `pipe_id`)
- Changed `time_shift` argument from boolean to integer


## Other Changes:
- Support FLUX.1
- Support Z-Image
- Support Krea2
- Allow stride > 2
- Add `cached_step` argument (e.g. 4 cached_step = skips every 4th step by reusing cached result)
