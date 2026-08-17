from judges import OllamaJudge

judge = OllamaJudge()

# a case where evidence clearly satisfies the control
r = judge.assess(
    control_text="Backups of data are created, protected, maintained, and tested",
    evidence="Backups are created on a schedule, encrypted, access-restricted, stored offsite, and restore-tested quarterly."
)
print("Should be MET:", r)

# a case where evidence is only topically related (should NOT be met)
r2 = judge.assess(
    control_text="Actions performed during an investigation are recorded, and the records' integrity and provenance are preserved",
    evidence="Suspected incidents are reported to the Security Operations team and managed through an incident-response process."
)
print("Should be GAP/PARTIAL:", r2)