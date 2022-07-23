import dsl


DSL_DICT = {
    ('list', 'list'): [dsl.MapFunction],
    ('list', 'atom'): [dsl.FoldFunction, dsl.SimpleITE],
    ('atom', 'atom'): [dsl.AntSimpleITE, dsl.mujocoant.AntUpPrimitiveFunction, dsl.mujocoant.AntDownPrimitiveFunction,
                       dsl.mujocoant.AntLeftPrimitiveFunction, dsl.mujocoant.AntRightPrimitiveFunction],
    ('atom', 'singleatom'): [dsl.AntPositionSelection, dsl.AntGoalPosSelection]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list'): {},
    ('list', 'atom'): {},
    ('atom', 'singleatom'): {},
    ('atom', 'atom'): {}
}
