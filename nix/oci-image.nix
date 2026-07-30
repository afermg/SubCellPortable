{
  pkgs,
  name,
  title,
  description,
  source,
  revision,
  server,
  entrypoint,
  port ? 5555,
  cmd ? ["tcp://0.0.0.0:${toString port}"],
  extraEnv ? [],
}:
pkgs.dockerTools.buildLayeredImage {
  name = "nahual/${name}";
  tag = "local";

  created = "1970-01-01T00:00:01Z";
  contents = [
    server
    pkgs.cacert
  ];
  extraCommands = ''
    mkdir -p tmp/nahual
    chmod 1777 tmp tmp/nahual
  '';

  config = {
    Entrypoint = [entrypoint];
    Cmd = cmd;
    ExposedPorts = {
      "${toString port}/tcp" = {};
    };
    WorkingDir = "/tmp/nahual";
    Env =
      [
        "HOME=/tmp/nahual"
        "USER=nahual"
        "LOGNAME=nahual"
        "PYTHONUNBUFFERED=1"
        "PYTHONDONTWRITEBYTECODE=1"
        "PYTHONSAFEPATH=1"
        "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
        "NVIDIA_VISIBLE_DEVICES=all"
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
        "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/run/opengl-driver/lib"
      ]
      ++ extraEnv;
    Labels = {
      "org.opencontainers.image.title" = title;
      "org.opencontainers.image.description" = description;
      "org.opencontainers.image.source" = source;
      "org.opencontainers.image.revision" = revision;
      "org.opencontainers.image.version" = "local";
      "org.opencontainers.image.created" = "1970-01-01T00:00:01Z";
      "org.opencontainers.image.vendor" = "Nahual";
    };
  };

  passthru = {
    imageName = "nahual/${name}";
    imageTag = "local";
  };
}
