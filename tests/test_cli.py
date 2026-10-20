import sys
import types

from inverser import cli


def test_azure_token_refresh_keeps_the_real_cli_login_after_home_isolation(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("AZURE_CONFIG_DIR", raising=False)
    seen = []

    class Credential:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def provider(credential, scope):
        # Mirrors the Azure CLI credential: each refresh resolves the config directory anew.
        return lambda: seen.append(cli.os.environ["AZURE_CONFIG_DIR"]) or "token"

    fake = types.ModuleType("azure.identity")
    fake.DefaultAzureCredential = Credential
    fake.get_bearer_token_provider = provider
    monkeypatch.setitem(sys.modules, "azure.identity", fake)

    def run(token):
        token()
        # What configure_physics_runtime() does once the first simulator is built.
        cli.os.environ["HOME"] = str(tmp_path / "isolated-pooltool-home")
        token()
        return {"status": "ok"}

    assert cli._with_azure_token(run) == {"status": "ok"}
    assert seen == [str(home / ".azure")] * 2
