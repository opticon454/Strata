import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { currentUserSid } from '../../server/private-file.mjs';

/** The effective protection of a file: mode 0600 on POSIX, on NTFS a protected DACL with exactly one full-control entry for the current user.
    The account is compared as the numeric SID read off the entry, not as the SDDL text: Get-Acl writes the built-in Administrator account's SID as
    LA in an SDDL, and a GitHub-hosted Windows runner runs the tests as that very account, so an SDDL comparison would compare one account's two spellings. */
export function assertPrivate(path) {
  if (process.platform !== 'win32') { assert.equal(statSync(path).mode & 0o777, 0o600); return; }
  const command = `$a=Get-Acl -LiteralPath '${path.replace(/'/g, "''")}'; "Protected|$($a.AreAccessRulesProtected)"; "Sddl|$($a.Sddl)"; foreach($e in $a.Access){ "Entry|$($e.AccessControlType)|$($e.FileSystemRights)|$($e.IsInherited)|$($e.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value)" }`;
  const lines = execFileSync(join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'), ['-NoProfile', '-NonInteractive', '-Command', command],
    { encoding: 'utf8', timeout: 30000, windowsHide: true }).split(/\r?\n/).filter(Boolean);
  const sddl = (lines.find(l => l.startsWith('Sddl|')) || '').slice(5);
  assert.equal(lines.find(l => l.startsWith('Protected|')), 'Protected|True', `ACL is not a protected owner-only DACL: ${sddl}`);
  const entries = lines.filter(l => l.startsWith('Entry|')).map(l => l.slice(6).split('|'));
  assert.equal(entries.length, 1, `ACL is not owner-only: it does not hold exactly one entry: ${sddl}`);
  const [type, rights, inherited, sid] = entries[0];
  assert.equal(`${type}/${inherited}/${sid}`, `Allow/False/${currentUserSid()}`, `ACL is not owner-only: ${sddl}`);
  const rightsSet = new Set(rights.split(', '));
  assert.ok(rightsSet.has('FullControl') && [...rightsSet].every(r => r === 'FullControl' || r === 'Synchronize'), `ACL is not the owner's full control: ${sddl}`);
}
