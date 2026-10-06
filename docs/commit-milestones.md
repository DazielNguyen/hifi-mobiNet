# Local commit milestones

These commits are new work, not reconstructed historical development.

| Commit | Milestone |
|---|---|
| `c5cc2c5` | Initialize independent repository, ignore rules and provenance policy |
| `bbed09f` | Import unchanged model/frontend/configuration references and notices |
| `b0c6897` | Add manifest-only inference, model access, adapted component namespace and contract tests |
| `bfd02da` | Add stored-result evaluation and portable statistical tools |
| `c028aff` | Preserve historical result tables, data manifests and aligned audio catalog |
| `2b6081e` | Add Streamlit UI, architecture/frontend tests and explicit local asset tools |
| `877b3bd` | Document setup, usage, deployment limits and release requirements |

The final verification commit adds the clean-checkout test record, source preservation
record and refreshed release/history audit. Its hash is available from `git log`;
this document cannot contain its own future commit hash.

At those initial milestones, no history was amended/squashed, no source repository
was committed and no remote was created. On 2026-10-06 the author subsequently
authorized public repository creation and code push.

| Later commit | Milestone |
|---|---|
| `1d8c7e7` | Verify clean checkout and audit release contents |
| `2fbea09` | Verify all four Q05 ONNX models, twelve local Harvard WAVs and complete Harvard inventory |

The subsequent publication-preparation commit records the real GitHub clone URL
and explicit author decisions. Weights/audio are not included in the push.
