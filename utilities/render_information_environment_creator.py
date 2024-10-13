from ray.tune.registry import _Registry, register_env

from environments.render_information_wrapper import RenderInformationWrapper


def register_render_information_environment_creator(environment_name: str):
    render_information_environment_name = environment_name + 'RenderInformation'

    def render_information_environment_creator(configuration: dict):
        environment_creator = _Registry().get('env_creator', environment_name)
        environment = environment_creator(configuration)
        render_information_environment = RenderInformationWrapper(environment)
        return render_information_environment

    register_env(name=render_information_environment_name, env_creator=render_information_environment_creator)

    return render_information_environment_name
