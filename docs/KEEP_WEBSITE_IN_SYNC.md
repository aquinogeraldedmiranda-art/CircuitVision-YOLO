# Keeping the CircuitVision website accurate as we develop it

CircuitVision is one ongoing student project and **one existing Render website**.
The About page reads `GET /api/project-status`, which combines two different
kinds of evidence:

1. **Reviewed project progress** in `project_status.json`. This is a versioned
   record of what we implemented, prepared, tested, and are awaiting results on.
2. **Actual runtime readiness** from `GET /api/health`. This records the
   model loaded by the deployed instance, its class count, whether the separate
   Round 3/Sam checkpoint files are available, and the deployed commit.

## After every meaningful project change

When adding a new trained model, frontend feature, backend change, dataset,
notebook, or camera improvement:

1. Make the code or notebook change in the **same** GitHub repository.
2. Update `project_status.json` in **the same pull request**. Adjust relevant
   milestones and add a concrete entry under `recent_updates`. Set
   `reviewed_on` to the actual review date, not a future date.
3. Distinguish **prepared**, **tested locally**, **deployed**, and
   **validated on real components**. A saved notebook, training plan or
   imported checkpoint does not prove a training run succeeded.
4. Preserve original checkpoint/version details and do not describe a
   visual resistor-value prediction as an electrical measurement.
5. Run tests. A GitHub Actions PR guard rejects feature changes that neglect
   the progress file; the status endpoint also has automated tests.
6. Merge only when tests pass and deploy the **same** Render service.
   Check the service's live deploy commit and `/api/project-status`.
7. Update the research paper only with matched methods, real dataset
   provenance, model versions, held-out testing and actual measurements.

GitHub and Render can automate **deployment of committed changes**, not the
truthfulness of unshared Colab training outcomes. If a notebook trains a new
candidate in a separate Colab session, the developer needs its exported
weights, training report, and validation results before marking it completed.

The frontend intentionally reads live status via the backend instead of
hardcoding temporary Round 2/Round 3/Sam claims in user-facing copy. This
allows the About page to reflect the active model without another frontend
rewrite when the final unified model is validated and activated.

## Official website

https://circuitvision-yolo-lab.onrender.com/#/about
