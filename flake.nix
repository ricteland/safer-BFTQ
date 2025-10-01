{
  description = "SRL Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    unstable-nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, unstable-nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        unstable-pkgs = import unstable-nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python312;
        pythonPackages = pkgs.python312.pkgs;

        cudatoolkit = pkgs.cudatoolkit;

        cudaEnvHook = ''
          export CUDA_HOME=${cudatoolkit}
          export CUDA_ROOT=${cudatoolkit}
          export LD_LIBRARY_PATH="${cudatoolkit.lib}/lib:${cudatoolkit}/lib:$LD_LIBRARY_PATH"
          export PATH="${cudatoolkit}/bin:$PATH"
          export CMAKE_PREFIX_PATH="${cudatoolkit}:$CMAKE_PREFIX_PATH"
        '';

        rl-agent-src = pkgs.fetchFromGitHub {
         owner = "eleurent";
         repo = "rl-agents";
         rev = "84df15ea977271e6a4d015f10f9f355f7e866890";
         sha256 = "sha256-EhLE0dEtlO4VnzA6u4CHlaxQNICKDIXS+n+LCjYrEA0=";
        };

        rl-agent= pythonPackages.buildPythonPackage {
          pname = "rl-agents";
          version = "18f2e7f";
          src = rl-agent-src;
         doCheck = false;

         nativeBuildInputs = with pythonPackages; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with pythonPackages; [
           gymnasium
           numpy
           pandas
           numba
           pygame
           matplotlib
           seaborn
           torch
           tensorboardx
           scipy
           six
           docopt
          ];
        };

        highway-src = pkgs.fetchFromGitHub {
         owner = "Farama-Foundation";
         repo = "HighwayEnv";
         rev = "75342a1b77e7ed33b99330a356890ffe31fbf9cb";
         sha256 = "sha256-gGs1Zd+MMBXRAkrFmvMGMBNNh0jTEDQ3x/+PLR2MHfg=";
        };

        highway = pythonPackages.buildPythonPackage {
          pname = "highway-env";
          version = "1.8.2";
          src = highway-src;
         doCheck = false;

         nativeBuildInputs = with pythonPackages; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with pythonPackages; [
           gymnasium
           numpy
           pygame
           matplotlib
           pandas
           scipy
          ];
        };

        mainPythonPackages = ps: with ps; [
          cython
          torch
          pyro-ppl
          gymnasium
          docopt
          numpy
          numba
          pandas
          pandas-stubs
          pygame
          matplotlib
          seaborn
          tensorboardx
          scipy
          pytest
          imageio
          moviepy
         six
         notebook
         pyvirtualdisplay
         highway
          tables.overrideAttrs (oldAttrs: {
            doCheck = false;
          })
        ];

        pythonEnv = python.withPackages mainPythonPackages;

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.ffmpeg
            pythonEnv
            pkgs.cmake
            pkgs.ninja
            cudatoolkit
            unstable-pkgs.gemini-cli
          ];

          shellHook = cudaEnvHook + ''
            echo "CUDA toolkit available at: $CUDA_HOME"
            echo "Python environment with Torch, CUDA. "
          '';
        };
      });
}
