{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "insurance-fraud-detection-env";

  buildInputs = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.fastapi
    pkgs.python311Packages.uvicorn
    pkgs.python311Packages.pandas
    pkgs.python311Packages.numpy
    pkgs.python311Packages.scikit-learn
    pkgs.python311Packages.xgboost
    pkgs.python311Packages.jinja2
    pkgs.python311Packages.python-multipart
    pkgs.python311Packages.imbalanced-learn
  ];

  shellHook = ''
    echo "🚀 Insurance Fraud Detection FastAPI Environment Ready"
    echo "Python version:"
    python --version
  '';
}
