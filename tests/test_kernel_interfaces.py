import tempfile
from pathlib import Path
import unittest

from musaprobe.kernel_debug import inspect_kernel_interfaces


class KernelInterfaceTests(unittest.TestCase):
    def test_compiler_headers_are_excluded_and_match_cap_does_not_end_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"clang"/"AST").mkdir(parents=True)
            (root/"clang"/"AST"/"Dump.h").write_text("void dumpAST();\n"*150)
            (root/"aaa.h").write_text("void dumpOther();\n"*150)
            (root/"musa_runtime.h").write_text("int musaDebugOutput();\n")
            (root/"zz_last.hpp").write_text("void final_header();\n")
            report=inspect_kernel_interfaces(root)
            self.assertEqual(report["scanned_files"],3)
            self.assertEqual(report["excluded_compiler_directories"],1)
            self.assertEqual(report["scanned_headers"][0]["file"],"musa_runtime.h")
            self.assertEqual(report["matches"][0]["file"],"musa_runtime.h")
            self.assertEqual(report["truncation"]["omitted_matches"],51)
            self.assertTrue(report["truncated"])
            self.assertFalse(report["supported_runtime_dump_verified"])

    def test_oversized_and_symlink_headers_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"musa.h").write_text("int musaLaunch();\n")
            (root/"big.h").write_bytes(b"x"*(2*1024**2+1))
            (root/"alias.h").symlink_to(root/"musa.h")
            report=inspect_kernel_interfaces(root)
            self.assertEqual(report["scanned_files"],1)
            self.assertEqual(report["excluded_symlinks"],1)
            self.assertEqual(report["truncation"]["oversized_files"],1)
            self.assertTrue(report["truncated"])


if __name__ == "__main__": unittest.main()
