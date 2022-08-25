import dsl


DSL_DICT = {
    ('atom', 'atom'): [dsl.POCAffine, dsl.POCITE1],
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list'): {},
    ('list', 'atom'): {},
    ('atom', 'singleatom'): {},
    ('atom', 'atom'): {}
}
