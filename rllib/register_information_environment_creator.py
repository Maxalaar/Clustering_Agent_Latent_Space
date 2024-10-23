from ray.tune.registry import _Registry, register_env

from environments.information_wrapper import InformationWrapper


def register_information_environment_creator(environment_name: str, save_rendering: bool = False):
    render_information_environment_name = environment_name + 'RenderInformation'

    def render_information_environment_creator(configuration: dict):
        environment_creator = _Registry().get('env_creator', environment_name)
        environment = environment_creator(configuration)
        render_information_environment = InformationWrapper(environment=environment, save_rendering=save_rendering)
        return render_information_environment

    register_env(name=render_information_environment_name, env_creator=render_information_environment_creator)

    return render_information_environment_name
