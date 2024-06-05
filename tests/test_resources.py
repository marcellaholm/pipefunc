import pytest

from pipefunc._resources import Resources


def test_valid_resources_initialization():
    res = Resources(
        num_cpus=4,
        num_gpus=1,
        memory="16GB",
        wall_time="2:00:00",
        queue="high",
        partition="gpu",
    )
    assert res.num_cpus == 4
    assert res.num_gpus == 1
    assert res.memory == "16GB"
    assert res.wall_time == "2:00:00"
    assert res.queue == "high"
    assert res.partition == "gpu"


def test_invalid_num_cpus():
    with pytest.raises(ValueError, match="num_cpus must be a positive integer."):
        Resources(num_cpus=-1)


def test_invalid_num_gpus():
    with pytest.raises(ValueError, match="num_gpus must be a non-negative integer."):
        Resources(num_gpus=-1)


def test_invalid_memory_format():
    with pytest.raises(
        ValueError,
        match=r"memory must be a valid string \(e\.g\., '2GB', '500MB'\).",
    ):
        Resources(memory="16XYZ")

    with pytest.raises(
        ValueError,
        match=r"memory must be a valid string \(e\.g\., '2GB', '500MB'\).",
    ):
        Resources(memory=1)


def test_invalid_wall_time_format():
    with pytest.raises(
        ValueError,
        match=r"wall_time must be a valid string \(e\.g\., '2:00:00', '48:00:00'\).",
    ):
        Resources(wall_time="invalid")


def test_from_dict():
    data = {
        "num_cpus": 4,
        "num_gpus": 1,
        "memory": "16GB",
        "wall_time": "2:00:00",
        "queue": "high",
        "partition": "gpu",
    }
    res = Resources.from_dict(data)
    assert res.num_cpus == 4
    assert res.num_gpus == 1
    assert res.memory == "16GB"
    assert res.wall_time == "2:00:00"
    assert res.queue == "high"
    assert res.partition == "gpu"


def test_to_slurm_options():
    res = Resources(num_cpus=4, memory="16GB", wall_time="2:00:00")
    assert res.to_slurm_options() == "--cpus-per-task=4 --mem=16GB --time=2:00:00"

    res = res.update(partition="gpu", num_gpus=2)
    assert (
        res.to_slurm_options()
        == "--cpus-per-task=4 --gres=gpu:2 --mem=16GB --time=2:00:00 --partition=gpu"
    )


def test_update():
    res = Resources(num_cpus=4, memory="16GB", wall_time="2:00:00")
    res = res.update(queue="high", num_gpus=2)
    assert res.queue == "high"
    assert res.num_gpus == 2
    assert (
        res.to_slurm_options()
        == "--cpus-per-task=4 --gres=gpu:2 --mem=16GB --time=2:00:00 --partition=high"
    )


def test_extra_args():
    res = Resources(extra_args={"test_arg": "value"})
    assert res.extra_args["test_arg"] == "value"
    assert res.to_slurm_options() == "--test_arg=value"


def test_combine_max_empty_list():
    combined_res = Resources.combine_max([])
    assert combined_res == Resources()


def test_combine_max_single_resource():
    res = Resources(num_cpus=4, memory="16GB", wall_time="2:00:00")
    combined_res = Resources.combine_max([res])
    assert combined_res == res


def test_combine_max_none_values():
    res1 = Resources(num_cpus=4, memory="16GB")
    res2 = Resources(num_gpus=1, wall_time="2:00:00")

    combined_res = Resources.combine_max([res1, res2])
    assert combined_res.num_cpus == 4
    assert combined_res.num_gpus == 1
    assert combined_res.memory == "16GB"
    assert combined_res.wall_time == "2:00:00"
    assert combined_res.queue is None
    assert combined_res.partition is None


def test_combine_max_extra_args():
    res1 = Resources(extra_args={"arg1": "value1"})
    res2 = Resources(extra_args={"arg2": "value2"})
    res3 = Resources(extra_args={"arg1": "value3", "arg3": "value4"})

    combined_res = Resources.combine_max([res1, res2, res3])
    assert combined_res.extra_args == {"arg1": "value1", "arg2": "value2", "arg3": "value4"}


def test_combine_max_multiple_resources():
    res1 = Resources(num_cpus=4, memory="16GB", wall_time="2:00:00")
    res2 = Resources(num_cpus=2, memory="32GB", wall_time="4:00:00", queue="high")
    res3 = Resources(num_gpus=1, memory="8GB", wall_time="1:00:00", partition="gpu")

    combined_res = Resources.combine_max([res1, res2, res3])
    assert combined_res == Resources(
        num_cpus=4,
        num_gpus=1,
        memory="32GB",
        wall_time="4:00:00",
        queue="high",
        partition="gpu",
    )


def test_combine_max_memory_units():
    res1 = Resources(memory="2GB")
    res2 = Resources(memory="1024MB")
    res3 = Resources(memory="0.5TB")

    combined_res = Resources.combine_max([res1, res2, res3])
    assert combined_res == Resources(memory="0.5TB")


def test_invalid_num_nodes():
    with pytest.raises(ValueError, match="num_nodes must be a positive integer."):
        Resources(num_nodes=0)

    with pytest.raises(ValueError, match="num_nodes must be a positive integer."):
        Resources(num_nodes=-1)


def test_invalid_num_cpus_per_node():
    with pytest.raises(ValueError, match="num_cpus_per_node must be a positive integer."):
        Resources(num_cpus_per_node=0)

    with pytest.raises(ValueError, match="num_cpus_per_node must be a positive integer."):
        Resources(num_cpus_per_node=-1)


def test_num_cpus_and_num_nodes_conflict():
    with pytest.raises(ValueError, match="num_nodes and num_cpus cannot be specified together."):
        Resources(num_cpus=4, num_nodes=2)


def test_num_cpus_per_node_without_num_nodes():
    with pytest.raises(
        ValueError,
        match="num_cpus_per_node must be specified with num_nodes.",
    ):
        Resources(num_cpus_per_node=4)


def test_update_method():
    # Create an initial Resources instance
    initial_resources = Resources(
        num_cpus=4,
        memory="16GB",
        wall_time="2:00:00",
        extra_args={"key1": "value1"},
    )

    # Update existing attributes and add new extra arguments
    updated_resources = initial_resources.update(
        num_cpus=8,
        memory="32GB",
        extra_args={"key2": "value2"},
        new_key="new_value",
    )

    # Check that the existing attributes are updated correctly
    assert updated_resources.num_cpus == 8
    assert updated_resources.memory == "32GB"
    assert updated_resources.wall_time == "2:00:00"

    # Check that the extra arguments are updated and new ones are added
    assert updated_resources.extra_args == {
        "key1": "value1",
        "key2": "value2",
        "new_key": "new_value",
    }

    # Check that the original resources instance is not modified
    assert initial_resources.num_cpus == 4
    assert initial_resources.memory == "16GB"
    assert initial_resources.extra_args == {"key1": "value1"}


def test_resources_wrong_args():
    with pytest.raises(TypeError, match="The following arguments are allowed"):
        Resources.from_dict({"num_cpus": 4, "wrong_arg": 1})


def test_num_cpus_per_node():
    r = Resources(num_cpus_per_node=1, num_nodes=1)
    assert r.num_cpus_per_node == 1
    assert r.to_slurm_options() == "--nodes=1 --cpus-per-node=1"
