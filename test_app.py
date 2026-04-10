"""
Platform Trial Meta-Analysis — Selenium test suite.
Tests: app loading, demo data, covariance structure, pooling,
non-concurrent adjustment, SVG rendering, export.
"""
import pytest
import time
import os
import json
import math
import subprocess
import signal
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixture: local HTTP server + Selenium Chrome
# ---------------------------------------------------------------------------

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence


@pytest.fixture(scope="session")
def server():
    port = _free_port()
    root = str(Path(__file__).resolve().parent)
    os.chdir(root)
    srv = HTTPServer(("127.0.0.1", port), _QuietHandler)
    t = Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/index.html"
    srv.shutdown()


@pytest.fixture(scope="session")
def driver(server):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})

    drv = None
    # Try Chrome first, then Edge
    for browser in ["chrome", "msedge"]:
        try:
            if browser == "chrome":
                drv = webdriver.Chrome(options=opts)
            else:
                from selenium.webdriver.edge.options import Options as EdgeOptions
                from selenium.webdriver.edge.service import Service as EdgeService
                eopts = EdgeOptions()
                eopts.add_argument("--headless=new")
                eopts.add_argument("--no-sandbox")
                eopts.add_argument("--disable-dev-shm-usage")
                eopts.add_argument("--disable-gpu")
                eopts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
                drv = webdriver.Edge(options=eopts)
            break
        except Exception:
            continue

    if drv is None:
        pytest.skip("No Chrome or Edge WebDriver available")

    drv.set_page_load_timeout(60)
    drv.implicitly_wait(5)
    drv.get(server)
    time.sleep(1)
    yield drv
    drv.quit()


def _js(driver, script):
    """Execute JS and return result."""
    return driver.execute_script(f"return {script}")


def _load_demo_and_analyze(driver, nca=False):
    """Load demo data and run analysis."""
    driver.execute_script("loadDemo()")
    time.sleep(0.3)
    if nca:
        driver.execute_script("document.getElementById('nca-toggle').checked = true")
    driver.execute_script("runAnalysis()")
    time.sleep(0.5)


# ===========================================================================
# Tests
# ===========================================================================

class TestAppLoad:
    def test_01_loads_without_js_errors(self, driver):
        """App loads without JavaScript errors."""
        logs = driver.get_log("browser")
        # Filter out favicon 404s (not actual JS errors)
        severe = [l for l in logs if l["level"] == "SEVERE"
                  and "favicon" not in l.get("message", "")]
        assert len(severe) == 0, f"JS errors: {severe}"

    def test_02_title(self, driver):
        assert "Platform Trial" in driver.title


class TestDemoData:
    def test_03_demo_loads(self, driver):
        """Demo data loads 4 platforms with 7 treatment-vs-control comparisons."""
        _load_demo_and_analyze(driver)
        results = _js(driver, "window._ptma.getResults()")
        assert results is not None
        platforms = _js(driver, "window._ptma.getPlatforms()")
        assert len(platforms) == 4  # RECOVERY, REMAP-CAP, SOLIDARITY, PANORAMIC

        # Count total treatment arms
        total_arms = sum(len(p["arms"]) for p in platforms)
        assert total_arms == 7  # 3 + 2 + 1 + 1


class TestCovariance:
    def test_04_shared_control_offdiag(self, driver):
        """For RECOVERY (3 arms), off-diagonal covariance = tau^2/2."""
        _load_demo_and_analyze(driver)
        results = _js(driver, "window._ptma.getResults()")
        tau2 = results["tau2"]

        # Find RECOVERY block
        recovery = None
        for b in results["blocks"]:
            if b["platform"] == "RECOVERY":
                recovery = b
                break
        assert recovery is not None
        assert recovery["K"] == 3

        # Off-diagonal should be tau^2 / 2
        expected_offdiag = tau2 / 2
        V = recovery["V"]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(V[i][j] - expected_offdiag) < 1e-10, \
                        f"V[{i}][{j}] = {V[i][j]}, expected {expected_offdiag}"

    def test_05_recovery_3x3_structure(self, driver):
        """RECOVERY var-cov matrix is 3x3 with correct structure."""
        _load_demo_and_analyze(driver)
        results = _js(driver, "window._ptma.getResults()")
        tau2 = results["tau2"]

        recovery = [b for b in results["blocks"] if b["platform"] == "RECOVERY"][0]
        V = recovery["V"]
        assert len(V) == 3
        assert len(V[0]) == 3

        # Diagonal: SE^2 + tau^2
        ses = [0.05, 0.06, 0.07]  # Demo data SEs
        for i in range(3):
            expected_diag = ses[i]**2 + tau2
            assert abs(V[i][i] - expected_diag) < 1e-10, \
                f"V[{i}][{i}] = {V[i][i]}, expected {expected_diag}"

    def test_06_independent_no_offdiag(self, driver):
        """Independent trials (SOLIDARITY, PANORAMIC) have no off-diagonal covariance."""
        _load_demo_and_analyze(driver)
        results = _js(driver, "window._ptma.getResults()")

        for name in ["SOLIDARITY", "PANORAMIC"]:
            block = [b for b in results["blocks"] if b["platform"] == name]
            assert len(block) == 1
            assert block[0]["K"] == 1  # Single arm = no off-diagonal


class TestPooling:
    def test_07_pooled_not_nan(self, driver):
        """Pooled effect accounting for covariance is not NaN."""
        _load_demo_and_analyze(driver)
        results = _js(driver, "window._ptma.getResults()")
        assert results["overall"]["beta"] is not None
        assert not math.isnan(results["overall"]["beta"])
        assert not math.isnan(results["overall"]["se"])

    def test_08_covariance_changes_result(self, driver):
        """Pooled CI differs when accounting for correlation vs ignoring it.

        When tau2=0 (DL estimate for demo data), off-diagonal=0 so GLS=naive.
        Force tau2>0 via manual mode to verify shared control covariance matters.
        """
        # Set manual tau2 to force non-zero off-diagonal covariance
        driver.execute_script("""
            document.getElementById('tau-mode').value = 'manual';
            document.getElementById('tau-manual').style.display = 'inline-block';
            document.getElementById('tau-manual').value = '0.01';
        """)
        _load_demo_and_analyze(driver)
        results_cov = _js(driver, "window._ptma.getResults()")
        beta_cov = results_cov["overall"]["beta"]
        se_cov = results_cov["overall"]["se"]
        tau2 = results_cov["tau2"]
        assert tau2 > 0, "Manual tau2 should be > 0"

        # Compute naive IV pooling (ignoring off-diagonal covariance)
        platforms = _js(driver, "window._ptma.getPlatforms()")
        sum_w = 0
        sum_we = 0
        for p in platforms:
            for arm in p["arms"]:
                v = arm["se"]**2 + tau2
                w = 1.0 / v
                sum_w += w
                sum_we += w * arm["effect"]
        beta_naive = sum_we / sum_w
        se_naive = math.sqrt(1.0 / sum_w)

        # With tau2=0.01, off-diagonal = 0.005, so GLS should differ from naive
        assert abs(se_cov - se_naive) > 1e-6 or abs(beta_cov - beta_naive) > 1e-6, \
            f"GLS (beta={beta_cov:.5f}, se={se_cov:.5f}) should differ from naive IV (beta={beta_naive:.5f}, se={se_naive:.5f})"

        # Reset to auto
        driver.execute_script("""
            document.getElementById('tau-mode').value = 'auto';
            document.getElementById('tau-manual').style.display = 'none';
        """)


class TestNCA:
    def test_09_nca_changes_estimates(self, driver):
        """Non-concurrent adjustment changes REMAP-CAP estimates (arms at different months)."""
        # Without NCA
        _load_demo_and_analyze(driver, nca=False)
        results_no_nca = _js(driver, "window._ptma.getResults()")
        remap_no = [b for b in results_no_nca["blocks"] if b["platform"] == "REMAP-CAP"][0]
        eff_no = remap_no["effects"][:]

        # With NCA
        driver.execute_script("document.getElementById('nca-toggle').checked = true")
        driver.execute_script("runAnalysis()")
        time.sleep(0.5)
        results_nca = _js(driver, "window._ptma.getResults()")
        remap_nca = [b for b in results_nca["blocks"] if b["platform"] == "REMAP-CAP"][0]
        eff_nca = remap_nca["effects"][:]

        # REMAP-CAP has arms starting at months 0 and 2 -> should differ
        assert eff_no != eff_nca, \
            f"NCA should change REMAP-CAP effects: no_nca={eff_no}, nca={eff_nca}"

        # Reset
        driver.execute_script("document.getElementById('nca-toggle').checked = false")

    def test_10_no_adjustment_when_concurrent(self, driver):
        """No adjustment when all arms are concurrent (same start/end)."""
        # SOLIDARITY has only 1 treatment arm starting at month 0 — no drift
        _load_demo_and_analyze(driver, nca=True)
        results = _js(driver, "window._ptma.getResults()")
        sol = [b for b in results["blocks"] if b["platform"] == "SOLIDARITY"][0]
        adj = sol.get("adjustments", [0])
        assert all(abs(a) < 1e-10 for a in adj), \
            f"SOLIDARITY (single arm) should have no adjustment: {adj}"
        driver.execute_script("document.getElementById('nca-toggle').checked = false")


class TestVisualization:
    def test_11_timeline_svg_renders(self, driver):
        """Timeline SVG renders with bars for each arm."""
        _load_demo_and_analyze(driver)
        svg = driver.execute_script(
            "return document.getElementById('timeline-container').querySelector('svg')"
        )
        assert svg is not None
        bars = driver.execute_script(
            "return document.getElementById('timeline-container').querySelectorAll('.gantt-bar').length"
        )
        # 4 controls + 7 treatments = 11 bars
        assert bars >= 7, f"Expected at least 7 bars, got {bars}"

    def test_12_forest_svg_rows(self, driver):
        """Forest plot SVG has correct number of study rows (squares)."""
        _load_demo_and_analyze(driver)
        svg = driver.execute_script(
            "return document.getElementById('forest-container').querySelector('svg')"
        )
        assert svg is not None
        # Should have squares for each treatment arm (7)
        squares = driver.execute_script(
            "return document.getElementById('forest-container').querySelectorAll('.forest-square').length"
        )
        assert squares == 7, f"Expected 7 study squares, got {squares}"

    def test_13_cov_matrix_display(self, driver):
        """Covariance matrix display renders for RECOVERY (multi-arm)."""
        _load_demo_and_analyze(driver)
        cov_html = driver.execute_script(
            "return document.getElementById('cov-container').innerHTML"
        )
        assert "RECOVERY" in cov_html
        # Should have off-diagonal cells
        assert "cov-offdiag" in cov_html
        assert "cov-diag" in cov_html


class TestEdgeCases:
    def test_14_single_platform_multiple_arms(self, driver):
        """k=1 platform with multiple arms gives valid result."""
        csv = """Platform,Arm,ArmType,Effect_vs_Control,SE,StartMonth,EndMonth,N
TRIAL1,DrugA,Treatment,-0.20,0.06,0,12,500
TRIAL1,DrugB,Treatment,-0.15,0.07,0,12,480
TRIAL1,Control,Control,0,0,0,12,500"""
        driver.execute_script(f"document.getElementById('csv-input').value = `{csv}`")
        driver.execute_script("runAnalysis()")
        time.sleep(0.5)
        results = _js(driver, "window._ptma.getResults()")
        assert results is not None
        assert not math.isnan(results["overall"]["beta"])
        assert len(results["blocks"]) == 1
        assert results["blocks"][0]["K"] == 2

    def test_15_two_arm_trial_degenerates(self, driver):
        """Two-arm trial degenerates to standard pairwise MA (1x1 block)."""
        csv = """Platform,Arm,ArmType,Effect_vs_Control,SE,StartMonth,EndMonth,N
SIMPLE,DrugX,Treatment,-0.10,0.05,0,12,1000
SIMPLE,Control,Control,0,0,0,12,1000"""
        driver.execute_script(f"document.getElementById('csv-input').value = `{csv}`")
        driver.execute_script("runAnalysis()")
        time.sleep(0.5)
        results = _js(driver, "window._ptma.getResults()")
        assert results is not None
        assert results["blocks"][0]["K"] == 1
        # No off-diagonal covariance
        V = results["blocks"][0]["V"]
        assert len(V) == 1
        assert len(V[0]) == 1


class TestSummaryAndExport:
    def test_16_summary_table_treatments(self, driver):
        """Summary table shows treatment-specific estimates."""
        _load_demo_and_analyze(driver)
        summary_html = driver.execute_script(
            "return document.getElementById('summary-container').innerHTML"
        )
        # Check all treatments present
        for trt in ["Dexamethasone", "Tocilizumab", "Baricitinib",
                     "Hydrocortisone", "Remdesivir", "Molnupiravir"]:
            assert trt in summary_html, f"{trt} missing from summary table"

        # Check heterogeneity stats present
        assert "tau" in summary_html
        assert "I" in summary_html

    def test_17_export_csv(self, driver):
        """Export CSV works (function callable without error)."""
        _load_demo_and_analyze(driver)
        # We can't test file download in headless, but we can verify the function runs
        error = driver.execute_script("""
            try {
                // Override download to capture content
                window._exportedCSV = null;
                const origBlob = window.Blob;
                window.Blob = function(content, opts) {
                    window._exportedCSV = content[0];
                    return new origBlob(content, opts);
                };
                exportCSV();
                window.Blob = origBlob;
                return null;
            } catch(e) {
                return e.message;
            }
        """)
        assert error is None, f"Export CSV error: {error}"
        csv_content = _js(driver, "window._exportedCSV")
        assert csv_content is not None
        assert "Treatment" in csv_content
        assert "Dexamethasone" in csv_content


class TestHeterogeneity:
    def test_18_tau2_nonnegative(self, driver):
        """tau^2 is non-negative."""
        _load_demo_and_analyze(driver)
        results = _js(driver, "window._ptma.getResults()")
        assert results["tau2"] >= 0

    def test_19_i2_range(self, driver):
        """I^2 is between 0 and 100."""
        _load_demo_and_analyze(driver)
        results = _js(driver, "window._ptma.getResults()")
        assert 0 <= results["I2"] <= 100


class TestMatrixOps:
    def test_20_matrix_inverse_identity(self, driver):
        """Matrix inverse * original = identity (2x2)."""
        result = driver.execute_script("""
            const m = [[4, 2], [1, 3]];
            const inv = window._ptma.invertMatrix(m);
            const prod = window._ptma.matMul(m, inv);
            // Check identity
            const ok = Math.abs(prod[0][0] - 1) < 1e-10 &&
                       Math.abs(prod[1][1] - 1) < 1e-10 &&
                       Math.abs(prod[0][1]) < 1e-10 &&
                       Math.abs(prod[1][0]) < 1e-10;
            return ok;
        """)
        assert result is True

    def test_21_matrix_inverse_3x3(self, driver):
        """Matrix inverse works for 3x3."""
        result = driver.execute_script("""
            const m = [[1, 0.5, 0.5], [0.5, 2, 0.5], [0.5, 0.5, 3]];
            const inv = window._ptma.invertMatrix(m);
            const prod = window._ptma.matMul(m, inv);
            let maxErr = 0;
            for (let i = 0; i < 3; i++)
                for (let j = 0; j < 3; j++)
                    maxErr = Math.max(maxErr, Math.abs(prod[i][j] - (i===j ? 1 : 0)));
            return maxErr < 1e-8;
        """)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
