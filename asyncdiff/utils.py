def get_ramped_time_shift(time_shift, shifted_steps, infer_step):
    if infer_step == 0:
        new_shift = time_shift
    else:
        lst = list(range(shifted_steps + 1))
        ind = lst.index(shifted_steps - infer_step)
        s = round(ind * time_shift / len(lst)) + 1
        new_shift = min(s, time_shift)
    return new_shift
