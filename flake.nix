{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/161120e886d7146b49bc335dcd116b68e1e3e82d";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    pynng-flake.url = "github:afermg/pynng";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        modelPackages = rec {
          subcell = pkgs.python3.pkgs.callPackage ./nix/subcell.nix {};
          pynng = pkgs.python3.pkgs.callPackage ./nix/pynng-local.nix {};
          nahual =
            (pkgs.python3.pkgs.callPackage (inputs.nahual-flake + "/nix/nahual.nix") {
              inherit pynng;
            }).overridePythonAttrs
            (_: {
              # This pinned nixpkgs has loguru 0.7.2; Nahual's >=0.7.3
              # declaration does not reflect APIs used by the server.
              dontCheckRuntimeDeps = true;
            });
        };
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          modelPackages.subcell
          modelPackages.nahual
          modelPackages.pynng
          pp.loguru
        ]);
        runSubcell = pkgs.writeScriptBin "nahual-subcell" ''
          #!${pkgs.bash}/bin/bash
          set -e
          export PYTHONPATH=${self}
          ${python_with_pkgs}/bin/python ${self}/ensure_model.py \
            --model-channels "''${2:-rybg}" \
            --model-type "''${3:-mae_contrast_supcon_model}"
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        subcellApp = {
          type = "app";
          program = "${runSubcell}/bin/nahual-subcell";
        };
      in
        with pkgs; rec {
          packages =
            modelPackages
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "subcell";
                title = "Nahual SubCell";
                description = "SubCell feature extraction served through Nahual";
                source = "https://github.com/afermg/SubCellPortable";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runSubcell;
                entrypoint = subcellApp.program;
              };
            };
          inherit python_with_pkgs;
          scripts.runSubcell = runSubcell;
          apps = rec {
            subcell = subcellApp;
            default = subcell;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              python3Packages.venvShellHook
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
            ];
            currentSystem = system;
            venvDir = "./.venv";
            postVenvCreation = ''unset SOURCE_DATE_EPOCH'';
            postShellHook = ''unset SOURCE_DATE_EPOCH'';
            shellHook = ''
              runHook venvShellHook
              export PYTHONPATH=${self}
            '';
          };
        }
    );
}
