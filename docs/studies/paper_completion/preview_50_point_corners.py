"""Render the alternative 50-point sparse fits without changing manuscript PDFs."""
import shutil
import count_history_manuscript as p


def main():
    output=p.OUT/'preview50';output.mkdir(exist_ok=True)
    p.FIG=output
    for mode in ('perfect','uncertain'):
        bundles=[p.prepare(n,mode,early_points=50) for n in (100,1000)]
        p.corners(mode,bundles)
        for _,meta,_ in bundles:
            for suffix in ('draws.npz','figure.json'):
                name=f"{meta['name']}_{suffix}"
                shutil.copy2(p.OUT/name,output/name)


if __name__=='__main__':main()
