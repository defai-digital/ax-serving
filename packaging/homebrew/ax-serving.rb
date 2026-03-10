# Formula for ax-serving — lives in the homebrew-ax-serving tap repo at:
#   https://github.com/automatosx/homebrew-ax-serving
#
# Install:
#   brew tap automatosx/ax-serving
#   brew install ax-serving
#
# This formula installs pre-built binaries (not built from source) because:
#   - ax-serving requires Apple Silicon M3+ and Xcode Metal toolchain
#   - Building from source on every user machine is impractical
#   - Pre-built binaries are signed and notarized by AutomatosX

class AxServing < Formula
  desc "High-performance LLM inference serving for Apple Silicon M3+"
  homepage "https://github.com/automatosx/ax-serving"
  version "1.1.0"
  license "AGPL-3.0-only"

  # Only Apple Silicon is supported
  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/automatosx/ax-serving/releases/download/v#{version}/ax-serving-v#{version}-aarch64-apple-darwin.tar.gz"
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
      ax-serving has been installed. Two binaries are available:

        ax-serving      — single-worker inference server
        ax-serving-api  — multi-worker API gateway (orchestrator)

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
  end
end
