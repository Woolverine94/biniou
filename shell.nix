{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
    python311
    openssl
    perl
    ffmpeg
    gperftools
    libGL
    libGLU
    glib
    xorg.libX11
    xorg.libXi
    xorg.libXmu
    xorg.libXext
    xorg.libXt
    xorg.libXfixes
    xorg.libXrender
    xorg.libXcursor
    xorg.libxcb
    xorg.libXinerama
    xorg.libXrandr
    xorg.libXcomposite
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.libGL
      pkgs.libGLU
      pkgs.xorg.libX11
      pkgs.glib
    ]}:$LD_LIBRARY_PATH
  '';
}
