from accelerators.speca import speca_cal_type, speca_init


def test_fixed_mask_uses_forward_denoising_index_and_exact_counts():
    refresh_mask = (True, True, False, True, False)
    cache, current = speca_init(
        num_steps=5,
        base_threshold=0.01,
        decay_rate=0.01,
        min_taylor_steps=1,
        max_taylor_steps=4,
        num_layers=1,
        check_layer=0,
        refresh_mask=refresh_mask,
    )

    observed = []
    for step_idx in range(5):
        current.step = 4 - step_idx
        speca_cal_type(cache, current)
        observed.append(current.type)
        assert cache.check is False

    assert observed == ["full", "full", "Taylor", "full", "Taylor"]
    assert cache.full_count == 3
    assert cache.taylor_count == 2


def test_adaptive_path_still_accepts_none_mask():
    cache, current = speca_init(
        num_steps=4,
        base_threshold=0.01,
        decay_rate=0.01,
        min_taylor_steps=1,
        max_taylor_steps=2,
        num_layers=1,
        check_layer=0,
    )
    current.step = 3
    speca_cal_type(cache, current)
    assert current.refresh_mask is None
    assert current.type in {"full", "Taylor"}
