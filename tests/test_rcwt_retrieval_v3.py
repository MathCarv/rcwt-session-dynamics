"""Offline read-policy contracts; fake token counts are not agent evidence."""

from __future__ import annotations

import ast
import copy
import importlib
import json
from pathlib import Path
import unittest

import rcwt_retrieval_v3 as retrieval
from rcwt_memory_v3 import compact_structured
from rcwt_retrieval_v3 import read_memory


def tokenize(text):
    # Deliberately fake, deterministic tokenization for contract tests only.
    return list(range((len(text.encode("utf-8")) + 3) // 4))


def observation(tool, **content):
    return {"source": "tool", "tool": tool, "content": content}


def facts(case="bill-A", account="bank-Z", merchant="shop-X", order="parcel-Q",
          amount=12003, status="active", payment="cleared", returned="accepted"):
    result = [observation("read_invoice", invoice_id=case, account_id=account,
                          merchant_id=merchant, order_id=order, total_cents=amount, currency="BRL"),
              observation("read_account", account_id=account, verification_status=status,
                          holder_merchant_id=merchant)]
    if payment is not None:
        result.append(observation("read_payment", invoice_id=case, status=payment))
    if returned is not None:
        result.append(observation("read_return", order_id=order, inspection_status=returned))
    return result


def stored(observations=None, receipt=None):
    return compact_structured(tokenize, "", json.dumps({
        "observations": facts() if observations is None else observations,
        "tool_result": receipt}), 256).text


def step(case="bill-A", operation="payout", observations=None, **extra):
    return {"task": {"case_id": case, "operation": operation},
            "observations": observations or [], **extra}


def read(text=None, public=None, **kwargs):
    return read_memory(stored() if text is None else text, step() if public is None else public,
                       tokenize=tokenize, **kwargs)


class RetrievalTests(unittest.TestCase):
    def test_empty_memory_and_absent_case_return_identical_empty_input(self):
        def forbidden(_):
            self.fail("Empty read must not tokenize or invent a hint")
        self.assertEqual(read_memory("", step(), tokenize=forbidden), "")
        self.assertEqual(read_memory("{}", step(), tokenize=forbidden), "")
        self.assertEqual(read_memory(stored(), step(case="unseen"), tokenize=forbidden), "")

    def test_card_has_named_facts_literal_target_and_public_join_ids(self):
        card = json.loads(read())
        self.assertEqual(card["case_id"], "bill-A")
        self.assertEqual(card["operation"], "payout")
        self.assertEqual(card["retained_evidence"], {
            "invoice_amount_cents": 12003, "account_status": "active", "ownership_match": "yes",
            "payment_status": "cleared", "return_status": "accepted", "operation_already_booked": "no_record"})
        self.assertEqual(card["retained_ids"], {"account_id": "bank-Z", "merchant_id": "shop-X",
                                                "holder_merchant_id": "shop-X", "order_id": "parcel-Q"})
        self.assertEqual(card["invalidated_fields"], [])
        self.assertNotIn("decision", card)
        self.assertNotIn("reason_code", card)

    def test_selection_is_literal_not_suffix_prefix_or_nearest_record(self):
        records = facts() + facts(case="bill-A-OTHER", account="other-bank", merchant="other-shop",
                                 order="other-parcel", amount=777)
        text = stored(records)
        self.assertEqual(json.loads(read(text))["retained_evidence"]["invoice_amount_cents"], 12003)
        self.assertEqual(json.loads(read(text, step(case="bill-A-OTHER")))["retained_evidence"]["invoice_amount_cents"], 777)
        self.assertEqual(read(text, step(case="bill")), "")
        self.assertNotIn("other-bank", read(text))

    def test_account_update_omits_stale_values_instead_of_competing_nulls(self):
        public = step(observations=[observation("read_account", account_id="bank-Z",
                                               verification_status="revoked", holder_merchant_id="NEW-HOLDER")])
        card = json.loads(read(public=public))
        self.assertEqual(set(card["invalidated_fields"]),
                         {"account_status", "ownership_match", "holder_merchant_id"})
        for field in ("account_status", "ownership_match"):
            self.assertNotIn(field, card["retained_evidence"])
        self.assertNotIn("holder_merchant_id", card["retained_ids"])
        self.assertEqual(card["retained_ids"]["account_id"], "bank-Z")
        self.assertNotIn("NEW-HOLDER", json.dumps(card))
        self.assertNotIn("revoked", json.dumps(card))

    def test_invoice_update_masks_all_old_dependent_links_and_facts(self):
        public = step(observations=[observation("read_invoice", invoice_id="bill-A",
                                               total_cents=999999, account_id="NEW", order_id="NEW",
                                               merchant_id="NEW")])
        card = json.loads(read(public=public))
        self.assertEqual(card["retained_ids"], {})
        self.assertEqual(card["retained_evidence"], {"payment_status": "cleared",
                                                     "operation_already_booked": "no_record"})
        self.assertEqual(set(card["invalidated_fields"]), {
            "invoice_amount_cents", "account_status", "ownership_match", "return_status",
            "account_id", "merchant_id", "holder_merchant_id", "order_id"})
        self.assertNotIn("NEW", json.dumps(card))
        self.assertNotIn("999999", json.dumps(card))

    def test_payment_and_return_invalidate_only_their_matching_source(self):
        public = step(observations=[observation("read_payment", invoice_id="bill-A", status="pending"),
                                   observation("read_return", order_id="parcel-Q", inspection_status="rejected")])
        card = json.loads(read(public=public))
        self.assertEqual(card["invalidated_fields"], ["payment_status", "return_status"])
        self.assertNotIn("payment_status", card["retained_evidence"])
        self.assertNotIn("return_status", card["retained_evidence"])
        self.assertEqual(card["retained_evidence"]["account_status"], "active")
        self.assertNotIn("pending", json.dumps(card))
        self.assertNotIn("rejected", json.dumps(card))

    def test_different_identity_or_source_cannot_invalidate_target(self):
        updates = [observation("read_account", account_id="bill-A", verification_status="revoked"),
                   observation("read_payment", invoice_id="other-invoice", status="pending"),
                   observation("read_return", order_id="other-order", inspection_status="rejected"),
                   observation("read_invoice", invoice_id="other-invoice", total_cents=0),
                   {"source": "message", "tool": "read_payment", "content": {"invoice_id": "bill-A"}}]
        self.assertEqual(read(public=step(observations=updates)), read())

    def test_new_values_and_new_links_do_not_affect_the_invalidation_card(self):
        for tool, primary, identifier in (("read_invoice", "invoice_id", "bill-A"),
                                          ("read_account", "account_id", "bank-Z"),
                                          ("read_payment", "invoice_id", "bill-A"),
                                          ("read_return", "order_id", "parcel-Q")):
            one = observation(tool, **{primary: identifier, "status": "cleared", "total_cents": 1,
                                       "verification_status": "active", "holder_merchant_id": "alpha",
                                       "inspection_status": "accepted"})
            two = observation(tool, **{primary: identifier, "status": "pending", "total_cents": 999999,
                                       "verification_status": "revoked", "holder_merchant_id": "beta",
                                       "inspection_status": "rejected"})
            with self.subTest(tool=tool):
                self.assertEqual(read(public=step(observations=[one])), read(public=step(observations=[two])))

    def test_update_order_duplicates_and_irrelevant_metadata_do_not_change_card(self):
        updates = [observation("read_account", account_id="bank-Z"),
                   observation("read_payment", invoice_id="bill-A"),
                   observation("read_invoice", invoice_id="bill-A")]
        expected = read(public=step(observations=updates))
        actual = read(public=step(observations=list(reversed(updates)) + updates,
                                 oracle={"decision": "approve"}, future={"answer": "refund"}))
        self.assertEqual(actual, expected)

    def test_bookings_use_retained_actual_invoice_and_operation_independently(self):
        receipt = {"tool": "record_decision", "accepted": True, "case_id": "bill-A",
                   "decision": "approve", "amount_booked_cents": 12003}
        text = stored(receipt=receipt)
        updates = [observation("read_invoice", invoice_id="bill-A", account_id="NEW")]
        payout = json.loads(read(text, step(observations=updates)))
        refund = json.loads(read(text, step(operation="refund", observations=updates)))
        self.assertEqual(payout["retained_evidence"]["operation_already_booked"], "yes")
        self.assertEqual(refund["retained_evidence"]["operation_already_booked"], "no_record")
        self.assertNotIn("operation_already_booked", payout["invalidated_fields"])

    def test_missing_facts_remain_unknown_not_borrowed_from_current_values(self):
        text = stored(facts(payment=None, returned=None))
        card = json.loads(read(text))
        self.assertEqual(card["retained_evidence"]["payment_status"], "unknown")
        self.assertEqual(card["retained_evidence"]["return_status"], "unknown")
        updated = json.loads(read(text, step(observations=[observation(
            "read_payment", invoice_id="bill-A", status="cleared")])))
        self.assertNotIn("payment_status", updated["retained_evidence"])

    def test_orphans_cannot_be_joined_using_new_current_invoice_links(self):
        text = stored([observation("read_account", account_id="orphan", verification_status="active",
                                   holder_merchant_id="new-shop")])
        public = step(observations=[observation("read_invoice", invoice_id="bill-A",
                                               account_id="orphan", merchant_id="new-shop")])
        self.assertEqual(read(text, public), "")

    def test_payment_only_case_retains_its_known_fact_and_explicit_unknowns(self):
        text = stored([observation("read_payment", invoice_id="bill-A", status="pending")])
        card = json.loads(read(text))
        self.assertEqual(card["retained_evidence"]["payment_status"], "pending")
        self.assertIsNone(card["retained_evidence"]["invoice_amount_cents"])
        self.assertEqual(card["retained_evidence"]["ownership_match"], "unknown")
        self.assertEqual(card["retained_ids"], {})

    def test_no_input_mutation_or_external_state_and_reload_is_reproducible(self):
        text = stored()
        public = step(observations=[observation("read_account", account_id="bank-Z")])
        original = copy.deepcopy(public)
        expected = read(text, public)
        read(stored(facts(case="elsewhere")), step(case="elsewhere"))
        importlib.reload(retrieval)
        self.assertEqual(retrieval.read_memory(text, public, tokenize=tokenize), expected)
        self.assertEqual(public, original)
        self.assertEqual(text, stored())

    def test_exact_token_cap_and_overflow_fail_closed_without_truncation(self):
        text = read()
        count = len(tokenize(text))
        self.assertEqual(read(budget=count), text)
        with self.assertRaisesRegex(ValueError, "exceeds token budget"):
            read(budget=count - 1)
        with self.assertRaisesRegex(ValueError, "exceeds token budget"):
            read_memory(stored(), step(), tokenize=lambda value: list(value.encode()), budget=20)

    def test_bad_arguments_tokenizer_and_serialized_state_fail_closed(self):
        for budget in (0, -1, True, 3.5, None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                read(budget=budget)
        for text in (None, "not JSON", '{"records":[],"records":[]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_memory(text, step(), tokenize=tokenize)
        for public in ({}, {"task": {}}, step(case=""), step(operation="execute"),
                       step(observations=[observation("read_payment", invoice_id=True)])):
            with self.subTest(public=public), self.assertRaises(ValueError):
                read(public=public)
        for tokens in ([True], [-1], ["token"], "not a list"):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                read_memory(stored(), step(), tokenize=lambda _: tokens)

    def test_imports_are_only_public_writer_and_standard_library(self):
        tree = ast.parse(Path(retrieval.__file__).read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertLessEqual(imports, {"__future__", "json", "typing", "rcwt_memory_v3"})


if __name__ == "__main__":
    unittest.main()
