import ast
import traceback

with open('tests/test_hunter.py', 'r', encoding='utf-8') as f:
    code = f.read()

class Transformer(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        func_id = getattr(node.func, 'id', None)
        if func_id in ('generate_signal', 'compute_composite_score'):
            # If already has MarketState skip
            if node.args and getattr(getattr(node.args[0], 'func', None), 'id', '') == 'MarketState':
                return node
            
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            
            defaults = {
                'current_price': ast.Constant(value=50000.0),
                'rsi': ast.Constant(value=50.0),
                'ls_ratio': ast.Constant(value=1.0),
                'whale_net_vol': ast.Constant(value=0.0),
                'regime': ast.Constant(value='CHOPPY'),
                'news_sentiment': ast.Constant(value='NEUTRAL'),
                'macd_histogram': ast.Constant(value=0.0),
                'bb_position': ast.Constant(value=0.5),
                'vwap_diff_pct': ast.Constant(value=0.0),
                'divergence': ast.Constant(value='NONE'),
                'funding_rate': ast.Constant(value=0.0),
                'open_interest_delta': ast.Constant(value=0.0),
                'rsi_slope': ast.Constant(value=0.0),
                'stoch_rsi': ast.Constant(value=50.0),
                'mtf_agreement': ast.Constant(value=0.0),
                'volume_confirm': ast.Constant(value=True),
                'near_resistance': ast.Constant(value=False)
            }
            use_comp = kwargs.pop('use_composite', None)
            
            # Build MarketState keywords
            ms_kwargs = []
            for k, default_node in defaults.items():
                ms_kwargs.append(ast.keyword(arg=k, value=kwargs.get(k, default_node)))
                
            ms_call = ast.Call(func=ast.Name(id='MarketState', ctx=ast.Load()), args=[], keywords=ms_kwargs)
            
            node.args = [ms_call]
            node.keywords = [ast.keyword(arg='use_composite', value=use_comp)] if use_comp else []
            return node
        return node

try:
    tree = ast.parse(code)
    tree = Transformer().visit(tree)
    new_code = ast.unparse(tree)

    if 'MarketState' not in new_code:
        new_code = new_code.replace('from analysis import (', 'from analysis import (\n    MarketState,')
        
    with open('tests/test_hunter.py', 'w', encoding='utf-8') as f:
        f.write(new_code)
    print("Successfully transformed test_hunter.py")
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
