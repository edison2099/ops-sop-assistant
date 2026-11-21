"""Create a small FICTIONAL sample SOP PDF for demos and tests.
Run: python scripts/make_sample_pdf.py  ->  data/sample/sample_ground_ops_sop.pdf

The content is invented for illustration only. It is not a real airline or authority procedure."""

import sys
from pathlib import Path

import pymupdf

PAGES = [
    """SAMPLE GROUND OPERATIONS SOP (FICTIONAL - FOR DEMO ONLY)

1 Purpose and Scope

This sample manual describes ground operations procedures for a fictional airline called Demo Air. It is used only to demonstrate the Ops SOP Assistant. The duty manager on shift is responsible for making sure all ground staff follow this manual.""",
    """2 Bird Strike Procedure

After a reported bird strike, the duty engineer must inspect the aircraft before the next departure. The inspection covers the engines, the wing leading edges, the windshield and the landing lights.

The duty manager must file a wildlife hazard report within 24 hours of the event. The report includes the time, the runway, the flight phase and the bird species if known.

If any damage is found, the aircraft is grounded until maintenance control releases it.""",
    """3 Fuel Spill Procedure

For a fuel spill larger than 2 square metres, stop all refuelling and stop passenger boarding immediately.

Call the airport fire service on extension 2222. Keep all vehicles and ground power units at least 15 metres away from the spill.

Boarding may resume only after the fire service declares the area safe.""",
    """4 Lightning and Thunderstorm Alert

When a lightning alert is issued within 5 km of the airport, all ramp activity must stop. Ground staff must move to the nearest building or vehicle.

Refuelling must not take place during an active lightning alert. Ramp work resumes 15 minutes after the last lightning strike within 5 km.""",
    """5 Delayed Flight Handling

When a departure is delayed by more than 2 hours, the station must offer passengers a meal voucher.

For delays longer than 5 hours, passengers may choose a full refund or a free change to another flight. The duty manager must update the flight status on the display boards every 30 minutes.""",
]


def make(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in PAGES:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontsize=11, fontname="helv")
    doc.save(out)
    doc.close()
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/sample_ground_ops_sop.pdf")
    print(f"Wrote {make(target)}")
