# Formula for ax-serving - mirrored into the homebrew-ax-serving tap repo at:
#   https://github.com/defai-digital/homebrew-ax-serving
#
# Install:
#   brew tap defai-digital/ax-serving
#   brew install ax-serving
#
# This formula installs pre-built binaries (not built from source) because:
#   - ax-serving requires Apple Silicon M3+ and Xcode Metal toolchain
#   - Building from source on every user machine is impractical
#   - Pre-built binaries are signed and notarized by DefAI Digital

class AxServing < Formula
  desc "Runtime-neutral inference gateway plus Apple Silicon compatibility tools"
  homepage "https://github.com/defai-digital/ax-serving"
  version "2.3.0"
  license "Apache-2.0"

  # Only Apple Silicon is supported
  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/defai-digital/ax-serving/releases/download/v#{version}/ax-serving-v#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    else
      odie "ax-serving requires Apple Silicon (M1/M2/M3+). Intel Macs are not supported."
    end
  end

  def install
    bin.install "bin/ax-serving"
    bin.install "bin/ax-serving-api"
    bin.install "bin/ax-servingctl"
    bin.install "bin/ax-runtime-agent"
    bin.install "bin/ax-thor-agent"
    bin.install "bin/ax-dynamo-adapter"
    bin.install "bin/ax-mac-cluster-adapter"

    # Install default configs to $(brew --prefix)/etc/ax-serving/
    (etc/"ax-serving").mkpath
    etc.install "config/backends.yaml" => "ax-serving/backends.yaml"
    etc.install "config/serving.yaml" => "ax-serving/serving.yaml"
    etc.install "config/dynamo-adapter.example.env" => "ax-serving/dynamo-adapter.example.env"
    etc.install "config/mac-cluster-adapter.example.env" => "ax-serving/mac-cluster-adapter.example.env"
    etc.install "config/mac-cluster-manifest.example.json" => "ax-serving/mac-cluster-manifest.example.json"
    etc.install "config/serving.mac-cluster.example.yaml" => "ax-serving/serving.mac-cluster.example.yaml"
    etc.install "config/compatibility-manifest.schema.json" => "ax-serving/compatibility-manifest.schema.json"
    etc.install "config/compatibility-manifest.example.json" => "ax-serving/compatibility-manifest.example.json"

    doc.install "README.md"
    doc.install "LICENSE"
    doc.install "NOTICE"
    doc.install "LICENSING.md"
    doc.install "TRADEMARKS.md"
  end

  # brew services start ax-serving — runs the runtime-neutral gateway
  service do
    run [opt_bin/"ax-serving-api"]
    keep_alive true
    working_dir var
    log_path    var/"log/ax-serving.log"
    error_log_path var/"log/ax-serving.log"
    environment_variables AXS_ALLOW_NO_AUTH: "true",
                          AXS_CONFIG: etc/"ax-serving/serving.yaml"
  end

  def caveats
    <<~EOS
      ax-serving has been installed. Seven binaries are available:

        ax-serving         — macOS embedded compatibility server
        ax-serving-api     — portable runtime-neutral gateway
        ax-servingctl      — portable operator client
        ax-runtime-agent   — generic runtime-node adapter
        ax-thor-agent      — deprecated runtime-node adapter alias
        ax-dynamo-adapter  — NVIDIA Dynamo execution-domain adapter
        ax-mac-cluster-adapter — experimental Mac cluster domain adapter

      Hybrid candidates run ax-serving-api as the gateway, register
      ax-runtime-agent in front of AX Engine, and place ax-dynamo-adapter
      in front of each independently qualified NVIDIA Dynamo domain.
      The Mac cluster adapter is a protocol/coordinator foundation and is not
      evidence of distributed AX Engine runtime support.
      Complete the PRD evidence gates before treating a deployment as
      production certified.

      Quick start (gateway, no runtime registered yet):
        AXS_ALLOW_NO_AUTH=true ax-serving-api &
        curl http://localhost:18080/health

      Run as a background service:
        brew services start ax-serving

      Set an API key (recommended in production):
        export AXS_API_KEY=your-secret-key

      Configuration file:
        #{etc}/ax-serving/serving.yaml

      Routing policy file:
        #{etc}/ax-serving/backends.yaml

      Logs (when running as a service):
        #{var}/log/ax-serving.log
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/ax-serving --version")
    assert_match version.to_s, shell_output("#{bin}/ax-serving-api --version")
    assert_match version.to_s, shell_output("#{bin}/ax-servingctl --version")
    assert_match version.to_s, shell_output("#{bin}/ax-runtime-agent --version")
    assert_match version.to_s, shell_output("#{bin}/ax-thor-agent --version")
    assert_match version.to_s, shell_output("#{bin}/ax-dynamo-adapter --version")
  end
end
