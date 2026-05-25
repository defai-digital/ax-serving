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
  desc "High-performance LLM inference serving for Apple Silicon M3+"
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

  # ax-serving spawns llama-server at runtime for all inference.
  # llama-server ships with the llama.cpp formula.
  depends_on "llama.cpp"

  def install
    bin.install "bin/ax-serving"
    bin.install "bin/ax-serving-api"
    bin.install "bin/ax-runtime-agent"
    bin.install "bin/ax-thor-agent"

    # Install default configs to $(brew --prefix)/etc/ax-serving/
    (etc/"ax-serving").mkpath
    etc.install "config/backends.yaml" => "ax-serving/backends.yaml"
    etc.install "config/serving.yaml" => "ax-serving/serving.yaml"
  end

  # brew services start ax-serving — runs the single-worker inference server
  service do
    run [
      opt_bin/"ax-serving", "serve",
      "--port", "18080",
      "--config", etc/"ax-serving/serving.yaml",
      "--routing-config", etc/"ax-serving/backends.yaml",
    ]
    keep_alive true
    working_dir var
    log_path    var/"log/ax-serving.log"
    error_log_path var/"log/ax-serving.log"
    environment_variables AXS_ALLOW_NO_AUTH: "true"
  end

  def caveats
    <<~EOS
      ax-serving has been installed. Four binaries are available:

        ax-serving        — single-worker inference server
        ax-serving-api    — multi-worker API gateway (orchestrator)
        ax-runtime-agent  — generic runtime-node adapter
        ax-thor-agent     — legacy Thor runtime-node adapter alias

      Quick start (single worker, no model yet):
        AXS_ALLOW_NO_AUTH=true ax-serving serve --port 18080 &
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
    assert_path_exists bin/"ax-runtime-agent"
    assert_path_exists bin/"ax-thor-agent"
  end
end
