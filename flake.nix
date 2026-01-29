{
  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs.url = "github:NixOS/nixpkgs/161120e886d7146b49bc335dcd116b68e1e3e82d";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    pynng-flake.url = "github:afermg/pynng";
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
        packages = {
          subcell = pkgs.python3.pkgs.callPackage ./nix/subcell.nix { };
          nahual = (inputs.nahual-flake.packages.${system}.nahual);
          pynng = (inputs.pynng-flake.packages.${system}.pynng);
        };
        scripts =
          let
            python_with_pkgs = python3.withPackages (pp: [
              packages.pynng
              packages.nahual
              packages.subcell
              pp.loguru
            ]);
          in
          {
            runSubcell = pkgs.writeScriptBin "run_subcell" ''
               #!${pkgs.bash}/bin/bash
               export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:${packages.nahual}/lib/python3.13/site-packages:${packages.pynng}/lib/python3.13/site-packages
               ${python_with_pkgs}/bin/python ${self}/server.py ''${1:-"ipc:///tmp/subcell.ipc"}
            '';
          };
        apps = rec {
          subcell = {
            type = "app";
            program = "${self.scripts.${stdenv.hostPlatform.system}.runSubcell}/bin/run_subcell";
          };
          default = subcell;
        };
        devShells = {
          default =
            let
              python_with_pkgs = (
                python3.withPackages (pp: [
                  packages.nahual
                  packages.pynng
                  packages.subcell
                  pp.loguru
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
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:${packages.nahual}/lib/python3.13/site-packages:${packages.pynng}/lib/python3.13/site-packages
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
