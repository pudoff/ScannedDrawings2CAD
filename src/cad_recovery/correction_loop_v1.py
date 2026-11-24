"""Retry transient database failures without abandoning durable correction leases."""
import time
import psycopg
from .web_correction_worker_v1 import process_once


def run(settings,*,process=process_once,sleep=time.sleep,stopped=lambda:False,log=print,label='Correction'):
    delay=1.
    while not stopped():
        try:worked=process(settings)
        except (psycopg.OperationalError,psycopg.InterfaceError) as error:
            # Connection errors may contain a DSN; log the class, never its message.
            log(f'{label} database unavailable ({type(error).__name__}); retry in {delay:g}s. Leases retained.',flush=True)
            sleep(delay);delay=min(30.,delay*2);continue
        delay=1.
        if not worked:sleep(.5)
