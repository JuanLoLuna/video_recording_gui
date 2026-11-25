#!/usr/bin/env python3
"""Thalamus-protocol clock server for measuring the satellite-to-laptop clock mapping.

Runs on the Windows laptop (Python 3.11; also works on Linux).  Needs `pip install thalamus-neuro grpcio`.

Usage:
  python satellite/clock_server.py [--host 0.0.0.0] [--port 50060] [--period-s 1.0]

It speaks the Thalamus gRPC protocol (insecure) and implements two rpcs; every other rpc answers
UNIMPLEMENTED (servicer base-class default):

  analog  Same shape as the Sleeve clock service (Jarl's server.exe): once per period an
          AnalogResponse with one span "Wall" whose ulong_data = [laptop system clock ns
          (time.time_ns())] and whose `time` = laptop steady clock ns (time.perf_counter_ns(),
          Windows QueryPerformanceCounter).  So the CURRENT method works unchanged.

  ping    Bidirectional stream.  The proto's Pong has only `id` and `payload`, so there is no
          spare field for server timestamps.  The server therefore echoes the client's payload
          and APPENDS a fixed 36-byte trailer (little-endian):

            offset  size  field
            0       4     magic  b'CLK1'
            4       8     instance_id  u64   random, chosen at server startup
            12      8     t2_recv_steady_ns  i64   perf_counter_ns when the Ping was received
            20      8     t3_send_steady_ns  i64   perf_counter_ns just before the Pong is built
            28      8     wall_ns  i64   time.time_ns() at send

          pong.payload = client_payload + trailer.  A client reads the LAST 36 bytes
          (parse_trailer in clock_logger.py).  A client that ignores the trailer still sees its
          own payload as the prefix.

Startup prints the instance ID and listen address; Ctrl+C stops cleanly.
"""
import argparse
import asyncio
import secrets
import struct
import sys
import time

TRAILER = struct.Struct("<4sQqqq")  # magic, instance_id, t2, t3, wall
MAGIC = b"CLK1"


def build_trailer(instance_id, t2, t3, wall_ns):
    return TRAILER.pack(MAGIC, instance_id, t2, t3, wall_ns)


def make_servicer(thalamus_pb2, thalamus_pb2_grpc, grpc, instance_id, period_s):
    class ClockServicer(thalamus_pb2_grpc.ThalamusServicer):
        async def analog(self, request, context):
            while True:
                wall = time.time_ns()
                steady = time.perf_counter_ns()
                resp = thalamus_pb2.AnalogResponse(
                    spans=[thalamus_pb2.Span(begin=0, end=1, name="Wall")],
                    ulong_data=[wall], is_ulong_data=True, time=steady)
                yield resp
                await asyncio.sleep(period_s)

        async def ping(self, request_iterator, context):
            async for req in request_iterator:
                t2 = time.perf_counter_ns()  # as early as possible after the message is delivered
                payload = bytes(req.payload)
                rid = req.id
                t3 = time.perf_counter_ns()  # as late as possible before the reply is built
                trailer = build_trailer(instance_id, t2, t3, time.time_ns())
                yield thalamus_pb2.Pong(id=rid, payload=payload + trailer)

    return ClockServicer()


async def serve(host, port, period_s):
    import grpc
    from thalamus import thalamus_pb2, thalamus_pb2_grpc
    instance_id = secrets.randbits(64)
    server = grpc.aio.server()
    thalamus_pb2_grpc.add_ThalamusServicer_to_server(
        make_servicer(thalamus_pb2, thalamus_pb2_grpc, grpc, instance_id, period_s), server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if not bound:
        print(f"cannot bind {host}:{port}", file=sys.stderr)
        return 1
    await server.start()
    print(f"clock_server instance_id={instance_id:016x} listening on {host}:{bound} "
          f"(analog period {period_s}s)", flush=True)
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop(grace=1)
        print("clock_server stopped", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50060)
    ap.add_argument("--period-s", type=float, default=1.0, help="analog message period (default 1.0)")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(serve(args.host, args.port, args.period_s))
    except KeyboardInterrupt:
        print("clock_server stopped (Ctrl+C)", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
