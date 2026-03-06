from django.core.management import call_command

try:
    from celery import shared_task
except Exception:  # pragma: no cover
    def shared_task(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@shared_task
def nightly_edgar_sync(limit=None):
    args = []
    if limit:
        args.append(f"--limit={int(limit)}")
    call_command("sync_edgar_nightly", *args)
