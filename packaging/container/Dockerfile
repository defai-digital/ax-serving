# syntax=docker/dockerfile:1.7

FROM rust:1.88-bookworm AS builder
WORKDIR /src

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo build --locked --release \
    -p ax-serving-cli --bin ax-serving-api --bin ax-servingctl \
    -p ax-thor-agent --bin ax-runtime-agent

FROM debian:bookworm-slim AS runtime-base
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 65532 --shell /usr/sbin/nologin ax-serving
USER 65532:65532
WORKDIR /home/ax-serving

FROM runtime-base AS gateway
COPY --from=builder /src/target/release/ax-serving-api /usr/local/bin/ax-serving-api
COPY --from=builder /src/target/release/ax-servingctl /usr/local/bin/ax-servingctl
EXPOSE 18080 19090
ENTRYPOINT ["/usr/local/bin/ax-serving-api"]

FROM runtime-base AS agent
COPY --from=builder /src/target/release/ax-runtime-agent /usr/local/bin/ax-runtime-agent
EXPOSE 18081
ENTRYPOINT ["/usr/local/bin/ax-runtime-agent"]
