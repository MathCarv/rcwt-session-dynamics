# RCWT-S Online v4: local validity review

This records the pre-result, read-only review of the frozen v4 design,
reference simulator, statistical analysis and offline replay. No confirmatory
outputs were inspected for this review, and no model calls were made. This
document contains no confirmatory scores or interim outcome assessment.

No implementation blocker was identified in the inspected paths. This is a
bounded code-and-protocol review, not proof of all possible correctness
properties, an external preregistration, or independent hardware attestation.

## What the paired comparison measures

Both policies start each paired episode with the same generated fixture and an
empty policy state. Each policy then owns its actual simulated ledger. The
reference checks prior bookings before the other public rules; an executed
monetary action can therefore change the reference for a later step. After
different actions, the two arms can legitimately have different later correct
answers. The endpoint measures accuracy of complete policy trajectories under
their own states, not responses against identical fixed labels at every step.
An earlier unsafe booking can make a later hold correct; accuracy must be read
alongside the separately recorded unsafe attempts and booked amounts.

References: [state-dependent reference](../src/rcwt_agent_env.py#L186),
[independent scoring](../src/rcwt_agent_env.py#L212),
[actual simulated execution](../src/rcwt_agent_env.py#L281), and
[registered endpoint](rcwt_online_v4_protocol.md#predeclared-confirmation-endpoints).

The intervention combines deterministic storage and organization of retained
and current public facts. It performs source/ID resolution, joins and derived
fact calculations that the summary arm leaves to the model. It does not choose
the action, learn a policy, update model weights or repair a generated answer.
Both arms use the same planning-then-JSON actor, task-bound schema and constant
ordered-rule guide. Any eventual difference belongs to this combined context
system conditional on that common actor, not to an isolated retention effect,
intrinsic model improvement, or autonomous recursive self-improvement.

References: [public context construction](../src/rcwt_context_v4.py#L60),
[shared guide](../src/rcwt_decision_v4.py#L55),
[task-only schema](../src/rcwt_decision_v4.py#L73),
[shared actor wrapper](../src/rcwt_online_v4.py#L54), and
[step execution and memory boundary](../src/rcwt_online_v4.py#L151).

## Statistical interpretation

The registered confirmation has 32 paired episodes, with eight steps per arm
and eight episodes per family. The statistical units are the 32 episode pairs,
not the 512 executed decisions. The percentile bootstrap resamples paired
episode differences 10,000 times. It is an unstratified episode bootstrap over
the balanced cohort, not a separate family-stratified procedure or an exact
finite-sample coverage guarantee. Family comparisons are descriptive.

The accuracy gate requires an observed mean difference of at least **10
percentage points** and a two-sided 95% interval whose lower endpoint is
strictly **above zero**. It does **not** require the lower endpoint to exceed
10 points; passing cannot be described as establishing a minimum 10-point
population effect with 95% confidence.

The two additional guards require no increase in observed unsafe monetary
attempts and no increase in unsafe fictional amounts actually booked. These
are aggregate descriptive checks, not a statistical noninferiority test,
zero-risk guarantee or production safety certification. They can coexist with
individual-case regressions. A numerical accuracy pass alone is insufficient.

References: [pair and cohort validation](../src/rcwt_analysis_v4.py#L49),
[bootstrap implementation](../src/rcwt_agent_analysis.py#L158), and
[accuracy and safety gates](../src/rcwt_analysis_v4.py#L119).

Fresh instances remain within the same four synthetic families and generator.
The design fixes one model, quantization, runtime profile and inference seed.
Its interval does not measure variation across inference seeds, models,
hardware, unseen task families or real customer traffic. Development informed
the candidate and common actor; development outcomes must not be pooled into
the confirmation or treated as independent proof.

## Resource comparability

Arm order is balanced within each family. Recorded tokens and generation time
include planning, final action generation and summary compaction. Step time
also includes local tokenization and deterministic context processing. There
is no model-memory call for the deterministic candidate, but its local work is
not free. API charge can be US$0 while total monetary cost remains unknown.

These are descriptive measurements on one local runtime. Load, thermal state,
cache behavior and execution order can affect timing. The accuracy/safety gate
neither requires lower latency nor establishes a universal speed improvement;
development and setup costs remain separate from confirmation totals.

References: [metering](../src/rcwt_online_v4.py#L200),
[balanced schedule](../src/rcwt_online_v4.py#L235), and
[resource comparison](../src/rcwt_analysis_v4.py#L152).

## Evidence and permissible conclusion

Offline replay checks source and protocol bindings, registered corpus and
order, exact actor requests, recorded completions, memory transitions,
simulator effects, grades and counters. It is evidence of internal consistency
with those recorded inputs. It does not independently attest GPU execution,
server clocks, absence of unlisted runs elsewhere, or external preregistration.
The freeze is a local hash-bound record, not an externally witnessed timestamp.

A final positive conclusion requires the complete registered confirmation,
successful evidence and report verification, and every predeclared gate. It
must identify the synthetic setting, shared actor, combined engineered context
intervention and descriptive safety limits. No claim about confirmed gain,
production readiness or CloudWalk customer outcomes follows from this review.

References: [protocol validation](../src/rcwt_online_v4.py#L396),
[offline replay](../src/rcwt_online_v4.py#L462),
[analysis limitations](../src/rcwt_analysis_v4.py#L203), and
[registered second and final development revision](rcwt_online_v4_revision_02.md).
