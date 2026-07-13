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
  version "2.2.0"
  license "AGPL-3.0-or-later"

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

    # Install default configs to $(brew --prefix)/etc/ax-serving/
    (etc/"ax-serving").mkpath
    etc.install "config/backends.yaml" => "ax-serving/backends.yaml"
    etc.install "config/serving.yaml" => "ax-serving/serving.yaml"

    doc.install "README.md"
    doc.install "LICENSE"
    doc.install "LICENSING.md"
    doc.install "LICENSE-COMMERCIAL.md"
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
      ax-serving has been installed. Five binaries are available:

        ax-serving         — macOS embedded compatibility server
        ax-serving-api     — portable runtime-neutral gateway
        ax-servingctl      — portable operator client
        ax-runtime-agent   — generic runtime-node adapter
        ax-thor-agent      — deprecated runtime-node adapter alias

      Hybrid candidates run ax-serving-api as the gateway and register
      ax-runtime-agent in front of AX Engine or a certified CUDA runtime.
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
  end
end
