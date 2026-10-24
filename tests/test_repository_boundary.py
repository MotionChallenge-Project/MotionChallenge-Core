import ast
from pathlib import Path
import unittest


class RepositoryBoundaryTests(unittest.TestCase):
    def test_core_is_independent_and_has_no_output_or_template_directory(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('results','templates','runs','motion-challenge-adapters'):
            self.assertFalse((root/name).exists(),name)
        for p in (root/'src/motion_challenge').rglob('*.py'):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node,ast.Import):names=[a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom):names=[node.module or '']
                else:continue
                self.assertFalse(any(n.startswith('motion_challenge_adapter') for n in names),str(p))


if __name__=='__main__':unittest.main()
