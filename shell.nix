{ pkgs ? import <nixpkgs> {} }: let
  # pkgs = import (builtins.fetchTarball {
  #   name = "nixos-unstable-2019-12-08";
  #   url = "https://github.com/nixos/nixpkgs/archive/cc6cf0a96a627e678ffc996a8f9d1416200d6c81.tar.gz";
  #   sha256="1srjikizp8ip4h42x7kr4qf00lxcp1l8zp6h0r1ddfdyw8gv9001";
  # }) {};
in
with pkgs; mkShell {

  buildInputs = [

    (python37.withPackages(a: with a; [

    jupyterhub jupyterlab flask imageio pylint
    ipython pytorch jupyter pandas numpy scikitimage pillow scipy pyyaml
    matplotlib cython tensorflow easydict munkres

    ]))

  ];

}
