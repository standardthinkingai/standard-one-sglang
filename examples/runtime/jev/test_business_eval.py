"""CPU tests for the diagnostic runner; no model accuracy is asserted here."""

import copy
import importlib.util
import io
import json
import math
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

MODULE = Path(__file__).with_name("business_eval.py")
SPEC = importlib.util.spec_from_file_location("business_eval", MODULE)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def fake_response(case, temperature=1.0, mistaken=False, permutations=1):
    """A fixture backend, deliberately independent of any real model."""
    answers = {}
    for key, question in case["questions"].items():
        names, kind, target = (
            bench.option_names(question),
            question["type"],
            case["expected"][key],
        )
        index = (
            (0 if target else 1)
            if kind == "noul"
            else (target if kind == "score" else names.index(target))
        )
        top = (index + 1) % len(names) if mistaken else index
        vectors = []
        for permutation in range(permutations):
            vectors.append(
                [-0.5 if i == top else -4.0 - permutation for i in range(len(names))]
            )
        distributions = [bench.normalize(vector, temperature) for vector in vectors]
        ps = [
            sum(row[i] for row in distributions) / permutations
            for i in range(len(names))
        ]
        answer = {
            "type": kind,
            "logprobs": [dict(zip(names, vector)) for vector in vectors],
        }
        if kind == "noul":
            answer["noul"] = ps[0]
        else:
            answer["probabilities"] = dict(zip(names, ps))
            if kind == "choice":
                answer["choice"] = names[max(range(len(ps)), key=ps.__getitem__)]
            else:
                answer["score"] = sum(i * p for i, p in enumerate(ps))
        answers[key] = answer
    return {
        "model": bench.DEFAULT_MODEL,
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


class BusinessEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.digest = bench.load_cases(bench.DEFAULT_CASES)

    def test_fixture_groups_are_disjoint_and_all_types_present(self):
        self.assertEqual(len(self.cases), 24)
        self.assertEqual(sum(len(c["questions"]) for c in self.cases), 72)
        groups = {
            split: {c["family"] for c in self.cases if c["split"] == split}
            for split in ("calibration", "eval")
        }
        self.assertFalse(groups["calibration"] & groups["eval"])
        for case in self.cases:
            self.assertEqual(
                {q["type"] for q in case["questions"].values()},
                {"choice", "noul", "score"},
            )
            self.assertEqual(case["provenance"]["kind"], "synthetic")

    def test_expected_labels_follow_explicit_policies(self):
        # Machine-check boundary values in the authored answer key, not LLM output.
        for case in self.cases:
            s, family = case["state"], case["family"]
            if family == "invoice_payment_policy":
                expected = (
                    "duplicate"
                    if s["이미_지급"]
                    else "documents"
                    if not s["증빙_첨부"]
                    else "pay",
                    not s["이미_지급"] and s["금액_원"] > 1000000,
                    2
                    if s["지급기한까지_일수"] < 0
                    else 1
                    if s["지급기한까지_일수"] == 0
                    else 0,
                )
            elif family == "support_channel_contract":
                expected = (
                    {
                        "이중청구": "billing",
                        "로그인실패": "technical",
                        "사용방법": "guide",
                    }[s["확정_유형"]],
                    s["상담원_요청"] or s["자동해결_실패횟수"] >= 3,
                    2
                    if s["응답제한_분"] <= 30
                    else 1
                    if s["응답제한_분"] <= 240
                    else 0,
                )
            elif family == "warehouse_dispatch_policy":
                expected = (
                    "stop"
                    if s["금지품목"]
                    else "wait_payment"
                    if not s["결제완료"]
                    else "dispatch",
                    not s["금지품목"]
                    and s["결제완료"]
                    and s["재고수량"] > 0
                    and s["마감전"],
                    2
                    if s["약속출고까지_시간"] < 0
                    else 1
                    if s["약속출고까지_시간"] <= 24
                    else 0,
                )
            elif family == "travel_expense_evidence":
                expected = (
                    {"M": "meal", "T": "transport", "S": "supplies"}[s["경비분류"]],
                    s["금액_원"] > 150000 and not s["관리자서명"],
                    {"세금계산서": 0, "전자영수증": 1, "없음": 2}[s["증빙형식"]],
                )
            elif family == "vendor_bank_verification":
                expected = (
                    "duplicate_close"
                    if s["동일번호_지급완료"]
                    else "bank_hold"
                    if s["계좌변경"] and not s["독립확인"]
                    else "payable",
                    s["계좌변경"] or s["거래처등록후_일수"] < 30,
                    min(2, s["필수문서_누락개수"]),
                )
            elif family == "incident_contract_matrix":
                expected = (
                    "oncall"
                    if s["장애코드"] == "C1"
                    else "privacy"
                    if s["개인정보사건"]
                    else "service_desk",
                    s["고객데이터_노출"] and s["대외통지_승인"],
                    0
                    if s["서비스중단_분"] == 0
                    else 1
                    if s["서비스중단_분"] < 15
                    else 2,
                )
            elif family == "cold_chain_returns":
                expected = (
                    "discard"
                    if s["기록최고온도_C"] > 8
                    else "inspect"
                    if s["봉인손상"]
                    else "restock",
                    s["약속시간후_배송"] or s["기록최고온도_C"] > 8,
                    0
                    if s["냉장고밖_시간"] == 0
                    else 1
                    if s["냉장고밖_시간"] <= 2
                    else 2,
                )
            elif family == "temporary_access_policy":
                expected = (
                    {
                        "시스템관리자": "security",
                        "예산담당": "finance",
                        "일반사용자": "it_support",
                    }[s["요청역할"]],
                    s["재직중"] and s["필수교육완료"] and s["승인티켓완료"],
                    2
                    if s["현재권한_만료까지_일수"] <= 0
                    else 1
                    if s["현재권한_만료까지_일수"] <= 7
                    else 0,
                )
            else:
                self.fail(f"missing oracle for {family}")
            self.assertEqual(tuple(case["expected"].values()), expected, case["id"])

    def test_loader_rejects_duplicate_and_family_leakage(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            first = copy.deepcopy(self.cases[0])
            second = copy.deepcopy(first)
            path.write_text(json.dumps(first) + "\n" + json.dumps(second))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                bench.load_cases(path)
            second["id"], second["split"] = "new-id", "eval"
            path.write_text(json.dumps(first) + "\n" + json.dumps(second))
            with self.assertRaisesRegex(ValueError, "leaks across splits"):
                bench.load_cases(path)

    def test_request_never_includes_answer_key(self):
        request = bench.request_for(self.cases[0], bench.DEFAULT_MODEL, 1.0)
        self.assertEqual(set(request), {"state", "model", "questions", "options"})
        self.assertEqual(
            request["options"], {"temperature": 1.0, "return_logprobs": True}
        )

    def test_exact_metrics_and_interpolated_percentile(self):
        rows = [
            {
                "type": "choice",
                "task": "test",
                "probabilities": [0.75, 0.25],
                "target_index": 0,
                "expected": "A",
                "predicted": "A",
            },
            {
                "type": "choice",
                "task": "test",
                "probabilities": [0.75, 0.25],
                "target_index": 1,
                "expected": "B",
                "predicted": "A",
            },
        ]
        metrics = bench.probability_metrics(rows)
        self.assertAlmostEqual(metrics["nll"], -math.log(0.75 * 0.25) / 2)
        self.assertAlmostEqual(metrics["brier"], 0.625)
        self.assertAlmostEqual(metrics["ece"], 0.25)
        self.assertAlmostEqual(bench.macro_f1(rows), 1 / 3)
        self.assertAlmostEqual(bench.percentile([10, 20], 0.95), 19.5)
        noul = [{"type": "noul", "probabilities": [0.8, 0.2], "target_index": 1}]
        self.assertAlmostEqual(bench.probability_metrics(noul)["brier"], 0.64)

    def test_response_requires_actual_zero_tokens_and_logprobs(self):
        case = self.cases[0]
        for count in (None, 1, False):
            response = fake_response(case)
            response["usage"]["output_tokens"] = count
            with self.assertRaisesRegex(ValueError, "integer zero"):
                bench.read_answers(case, response)
        response = fake_response(case)
        del response["answers"]["route"]["logprobs"]
        with self.assertRaisesRegex(ValueError, "logprobs are absent"):
            bench.read_answers(case, response)

    def test_response_rejects_wrong_score_and_invalid_distribution(self):
        case = self.cases[0]
        response = fake_response(case)
        response["answers"]["deadline"]["score"] = 0
        with self.assertRaisesRegex(ValueError, "score disagrees"):
            bench.read_answers(case, response)
        response = fake_response(case)
        response["answers"]["route"]["probabilities"]["pay"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            bench.read_answers(case, response)

    def calibration_report(self):
        selected = [c for c in self.cases if c["split"] == "calibration"]
        index = 0

        def transport(url, request, timeout):
            nonlocal index
            response = fake_response(
                selected[index], mistaken=index % 4 == 0, permutations=2
            )
            index += 1
            return response

        return bench.run_evaluation(
            self.cases,
            self.digest,
            "http://fixture",
            bench.DEFAULT_MODEL,
            "calibration",
            1.0,
            transport=transport,
        )

    def test_calibration_uses_raw_permutations_and_reduces_nll(self):
        report = self.calibration_report()
        fitted = bench.fit_temperature(report, self.cases, self.digest)
        self.assertEqual(fitted["n_questions"], 36)
        self.assertLess(fitted["nll_after"], fitted["nll_before"])
        self.assertGreater(fitted["temperature"], 1.0)
        # Independent reconstruction: temperature acts before averaging rows.
        losses = []
        for trace in report["traces"]:
            for row in trace["observations"]:
                ps = [
                    bench.normalize(vector, fitted["temperature"])[row["target_index"]]
                    for vector in row["logprobs"]
                ]
                losses.append(-math.log(sum(ps) / len(ps)))
        self.assertAlmostEqual(fitted["nll_after"], sum(losses) / len(losses))

    def test_calibration_rejects_eval_and_modified_or_incomplete_inputs(self):
        report = self.calibration_report()
        wrong = copy.deepcopy(report)
        wrong["split"] = "eval"
        with self.assertRaisesRegex(ValueError, "never eval"):
            bench.fit_temperature(wrong, self.cases, self.digest)
        wrong = copy.deepcopy(report)
        wrong["traces"][0]["id"] = next(
            c["id"] for c in self.cases if c["split"] == "eval"
        )
        with self.assertRaisesRegex(ValueError, "non-calibration"):
            bench.fit_temperature(wrong, self.cases, self.digest)
        wrong = copy.deepcopy(report)
        wrong["traces"].pop()
        with self.assertRaisesRegex(ValueError, "every calibration"):
            bench.fit_temperature(wrong, self.cases, self.digest)
        wrong = copy.deepcopy(report)
        wrong["traces"][0]["request"]["state"] = "changed input"
        with self.assertRaisesRegex(ValueError, "differs from the fixture"):
            bench.fit_temperature(wrong, self.cases, self.digest)

    def test_failures_are_preserved_in_denominator_and_trace(self):
        selected = [c for c in self.cases if c["split"] == "eval"]

        def unavailable(*args):
            raise TimeoutError("fixture timeout")

        report = bench.run_evaluation(
            self.cases,
            self.digest,
            "http://fixture",
            bench.DEFAULT_MODEL,
            "eval",
            1.0,
            transport=unavailable,
        )
        summary = report["summary"]
        self.assertEqual(summary["failed_requests"], len(selected))
        self.assertEqual(summary["metrics"]["choice"]["accuracy"], 0)
        self.assertIsNone(summary["metrics"]["choice"]["nll"])
        self.assertFalse(summary["zero_output_tokens_verified"])
        self.assertIn("fixture timeout", report["traces"][0]["error"])

    def test_real_http_serialization_and_endpoint_path(self):
        case = self.cases[0]
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                received.append((self.path, payload, self.headers.get("Authorization")))
                body = json.dumps(fake_response(case)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.dict(bench.os.environ, {"SGLANG_API_KEY": "fixture-test-key"}):
                report = bench.run_evaluation(
                    [case],
                    self.digest,
                    f"http://127.0.0.1:{server.server_port}",
                    bench.DEFAULT_MODEL,
                    "calibration",
                    1.0,
                )
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
        self.assertEqual(received[0][0], "/v1/systemone")
        self.assertEqual(received[0][1]["state"], case["state"])
        self.assertEqual(received[0][2], "Bearer fixture-test-key")
        self.assertNotIn("fixture-test-key", json.dumps(report))
        self.assertEqual(report["summary"]["successful_requests"], 1)
        self.assertTrue(report["summary"]["zero_output_tokens_verified"])

    def test_validate_only_does_not_claim_model_performance(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(bench.main(["--validate-only"]), 0)
        data = json.loads(output.getvalue())
        self.assertFalse(data["model_performance_measured"])
        self.assertNotIn("metrics", data)
        self.assertEqual(data["cases"], 24)


if __name__ == "__main__":
    unittest.main()
