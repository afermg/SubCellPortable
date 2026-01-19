{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nixpkgs_transformers_2_5_1.url = "github:NixOS/nixpkgs/161120e886d7146b49bc335dcd116b68e1e3e82d"; # transformers 2.5.1
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    # nixpkgs.url = "github:NixOS/nixpkgs/d97b37430f8f0262f97f674eb357551b399c2003";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        libList = [
          pkgs.stdenv.cc.cc
          pkgs.stdenv.cc
          pkgs.libGL
          pkgs.gcc
          pkgs.glib
          pkgs.libz
          pkgs.glibc
        ];
      in
      with pkgs;
      rec {
        py312 = (
          pkgs.python312.override {
            packageOverrides = _: super: {
              transformers = super.transformers.overridePythonAttrs (old: rec {
						    version = "4.45.1";
						    doCheck = false;
						src = super.fetchPypi {
							pname = "transformers";
							inherit version;
							hash = "sha256-nKzhEHIXLfBcpqaU/NH1BkpVtjKF5JK9iPCtHOwnDwI=";
						};
					    });
            };
          }
        );
        packages = {
          subcell = py312.pkgs.callPackage ./nix/subcell.nix { };
        };
        devShells = {
          default =
            let
              python_with_pkgs = (
                python312.withPackages (pp: [
                  # (inputs.nahual-flake.packages.${system}.nahual)
                  packages.subcell
                  # packages.pynng
                ])
              );
            in
            mkShell {
              packages = [
                python_with_pkgs
                python3Packages.venvShellHook
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              currentSystem = system;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                # Set PYTHONPATH to only include the Nix packages, excluding current directory
                runHook venvShellHook
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}
              '';
            };
        };
      }
    );
}
# export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
# export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
# export NVCC_APPEND_FLAGS="-Xcompiler -fno-PIC"
# export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
# export CUDA_NVCC_FLAGS="-O2 -Xcompiler -fno-PIC"
# # Ensure current directory is not in Python path
# export PYTHONDONTWRITEBYTECODE=1
