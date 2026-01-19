{
  lib,
  # build deps
  buildPythonPackage,
  fetchFromGitHub,
  # Py build
  setuptools,
  # Deps
  torch,
  transformers,
  jupyter,
  torchvision,
  boto3,
}:
buildPythonPackage {
  pname = "subcell";
  version = "0.0.1";

  src = ./..; # For local testing, add flag --impure when running
  # src = fetchFromGitHub {
  #   owner = "afermg";
  #   repo = "SubCellPortable";
  #   rev = "9d3c372ace5ae5b0b6677933f619b6d48988f2ef";
  #   sha256 = "";
  # };

  pyproject = true;
  buildInputs = [
    # setuptools-scm
    setuptools
  ];
  propagatedBuildInputs = [
    torch
    transformers
    jupyter
    torchvision
    boto3
  ];

  pythonImportsCheck = [
  ];

  meta = {
    description = "Generic environment for visual transformers.";
    homepage = "https://github.com/afermg/nahual_vit";
    license = lib.licenses.mit;
  };
}
