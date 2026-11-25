"""Headless and same-origin Web use the same safe upload storage."""
from cad_recovery.service_profiles_v23 import api as previous
from cad_recovery.service_profiles_v10.storage import Storage

check_assets=previous.check_assets


def create_api(settings,**dependencies):
    dependencies.setdefault('storage',Storage(settings))
    return previous.create_api(settings,**dependencies)


def create_web_app(settings,**dependencies):
    dependencies.setdefault('storage',Storage(settings))
    return previous.create_web_app(settings,**dependencies)
