#!/usr/bin/env bash
# Regenerate Python protobuf stubs from proto/ax_serving.proto.
# Run from the repository root: bash sdk/python/scripts/gen_proto.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/sdk/python/ax_serving/_proto"

python3 -m grpc_tools.protoc \
  -I "$REPO_ROOT/proto" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$REPO_ROOT/proto/ax_serving.proto"

# Fix absolute import to relative in the generated grpc stub.
sed -i.bak 's/^import ax_serving_pb2 as ax__serving__pb2/from . import ax_serving_pb2 as ax__serving__pb2/' \
  "$OUT_DIR/ax_serving_pb2_grpc.py"
rm -f "$OUT_DIR/ax_serving_pb2_grpc.py.bak"

echo "Proto stubs regenerated in $OUT_DIR"
