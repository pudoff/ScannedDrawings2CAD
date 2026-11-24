"""Public producer dispatch, avoiding imports of executable build scripts."""


def build_table(request, progress):
    from .web_tables_contract_v1 import binding as first
    from .web_tables_contract_v2 import binding as second
    from .web_tables_contract_v3 import binding as third
    from .web_tables_contract_v4 import binding as fourth
    sha = request['table_binding']['sha256']
    if sha == fourth()['sha256']:
        from .web_tables_build_v4 import build
        phase = 'text_only_choice' if request['edit'].get('text_selection') else 'table_OCR_evidence'
        with progress.phase(phase):
            return build(request, progress=progress)
    if sha == first()['sha256']:
        from .web_tables_build_v1 import build
    elif sha == second()['sha256']:
        from .web_tables_build_v2 import build
    elif sha == third()['sha256']:
        from .web_tables_build_v3 import build
        with progress.phase('table_text_choice' if request['edit'].get('text_selection') else 'table_build'):
            return build(request, progress=progress)
    else:
        raise ValueError('Unsupported table producer')
    with progress.phase('table_build'):
        return build(request)
