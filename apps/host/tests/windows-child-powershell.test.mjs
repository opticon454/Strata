import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

/* Issue #4: the test run's Windows PowerShell 5.1 child must autoload Microsoft.PowerShell.Security and run Get-Acl, the way
   tests/helpers/assert-private.mjs does when it checks a saved key's ACL. The child here starts with the test run's own environment,
   so this test goes red exactly when that helper goes red. The probe answers with what decided the outcome: the PSModulePath the
   child was given, every Microsoft.PowerShell.Security it can find on it, and the full exception chain of a direct Import-Module. */
const probe = [
  '"Shell: " + $PSVersionTable.PSVersion',
  '"PSModulePath: " + $env:PSModulePath',
  'try { $null = Get-Acl -LiteralPath $env:TARGET -ErrorAction Stop; "GetAcl: OK" } catch { "GetAcl: " + $_.Exception.Message }',
  'foreach ($m in @(Get-Module -ListAvailable Microsoft.PowerShell.Security)) { "Found: " + $m.Path + " (v" + $m.Version + ", PS " + ($m.CompatiblePSEditions -join " ") + ")" }',
  'try { Import-Module Microsoft.PowerShell.Security -ErrorAction Stop; "Import: OK" } catch { for ($e = $_.Exception; $e; $e = $e.InnerException) { "ImportFail: " + $e.GetType().Name + ": " + $e.Message } }',
].join('\n');

test('the test run powers the Windows PowerShell 5.1 child so Get-Acl loads (the way #4 failed, it did not)',
  { skip: process.platform !== 'win32' && 'a Windows PowerShell child is a Windows thing (Windows only)' }, () => {
    const shell = join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
    const directory = mkdtempSync(join(tmpdir(), 'strata-ps-child-'));
    let out;
    try {
      out = execFileSync(shell, ['-NoProfile', '-NonInteractive', '-Command', probe],
        { encoding: 'utf8', timeout: 120000, windowsHide: true, env: { ...process.env, TARGET: directory } });
    } catch (error) { /* the probe only prints; a child that cannot start at all lands here */
      assert.fail(`the Windows PowerShell 5.1 child did not answer: ${error.message}\n${error.stdout || ''}`);
    } finally { rmSync(directory, { recursive: true, force: true }); }
    assert.match(out, /GetAcl: OK/, out);
    assert.match(out, /Import: OK/, out);
  });
