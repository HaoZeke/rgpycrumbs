import pytest
from click.testing import CliRunner

from tests.conftest import skip_if_not_env

skip_if_not_env("eon")

from rgpycrumbs.eon.plt_neb import main as plt_neb_main  # noqa: E402

pytestmark = pytest.mark.eon


@pytest.fixture
def runner():
    return CliRunner()


def test_plt_neb_strip_renderer_option(runner):
    """Verify --strip-renderer appears in plt-neb help output."""
    result = runner.invoke(plt_neb_main, ["--help"])
    assert result.exit_code == 0
    assert "--strip-renderer" in result.output
