def pytest_addoption(parser):
    parser.addoption("--require-cuda", action="store_true",
                     help="Fail rather than skip when the CUDA device is unavailable")
