#!/usr/bin/env node
/** Enumerate test files explicitly so npm test works without shell glob expansion. */
import {readdirSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const files=readdirSync(resolve(root,'tests')).filter(name=>name.endsWith('.test.mjs')).sort().map(name=>resolve(root,'tests',name));
/* Every test file runs against scratch copies of the per-user places the host writes to (the owner's real profile is never touched). */
import {mkdtempSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const scratch=mkdtempSync(join(tmpdir(),'strata-host-suite-profile-'));
const env={...process.env,STRATA_SETTINGS_FILE:join(scratch,'app-settings.json'),STRATA_COMPONENTS_DIR:join(scratch,'components'),LOCALAPPDATA:join(scratch,'Local'),APPDATA:join(scratch,'Roaming'),XDG_DATA_HOME:join(scratch,'xdg-data'),XDG_STATE_HOME:join(scratch,'xdg-state'),XDG_CONFIG_HOME:join(scratch,'xdg-config')};
/* Run from PowerShell 7 (GitHub's Windows runners), PSModulePath names PowerShell 7's module folders first and a Windows PowerShell 5.1 child
   that a test starts loads Get-Acl / Get-FileHash from them and cannot (#4: "The member Sddl is already present"). Keep only the folders
   that are not PowerShell 7's. The variable can reach this process under a different letter case than the tests ask for it by, and a plain
   object spread is case-sensitive: find the key whatever its case, and leave exactly one PSModulePath behind, so no stale copy survives. */
if(process.platform==='win32'){
  const seen=Object.keys(env).filter(k=>k.toLowerCase()==='psmodulepath');
  const ps7=p=>{const l=p.toLowerCase().replace(/\//g,'\\');return l.includes('\\powershell\\')&&!l.includes('\\windowspowershell\\');};
  const raw=env[seen[0]];
  for(const k of seen)delete env[k];
  if(raw)env.PSModulePath=raw.split(';').filter(p=>p&&!ps7(p)).join(';');
  console.error(`[host-test][#4] psmodulepath keys at start: ${JSON.stringify(seen)}; passed to the tests: ${JSON.stringify(env.PSModulePath)}`);
}
/* A test that never finishes fails after STRATA_TEST_TIMEOUT_MS (default 10 minutes) instead of holding the whole run for ever; extra arguments (a reporter, --test-name-pattern) go to node --test. */
const timeout=Number(process.env.STRATA_TEST_TIMEOUT_MS)||600000;
const child=spawn(process.execPath,['--test',`--test-timeout=${timeout}`,...process.argv.slice(2),...files],{cwd:root,stdio:'inherit',env});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',(code,signal)=>{process.exitCode=code??(signal?1:0);});
