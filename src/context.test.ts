import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setDbPath, closeDb, getAllNodes, findOrCreateNode } from './graph';
import { contextLabel, isContextNode, detectContext, getActiveContext, ensureContextNode, bindToContext } from './context';
import { execSync } from 'child_process';

vi.mock('child_process', () => ({
  execSync: vi.fn(),
}));

describe('context', () => {
  beforeEach(() => {
    setDbPath(':memory:');
    vi.clearAllMocks();
  });

  afterEach(() => {
    closeDb();
  });

  it('should format context label correctly', () => {
    expect(contextLabel('test')).toBe('[ctx:test]');
  });

  it('should identify context nodes correctly', () => {
    expect(isContextNode('[ctx:test]')).toBe(true);
    expect(isContextNode('test')).toBe(false);
  });

  it('should detect context from git remote', () => {
    vi.mocked(execSync).mockReturnValueOnce('https://github.com/user/my-repo.git\n');
    expect(detectContext()).toBe('project:my-repo');
  });

  it('should detect context from git toplevel if remote fails', () => {
    vi.mocked(execSync)
      .mockImplementationOnce(() => { throw new Error(); }) // remote fails
      .mockReturnValueOnce('/Users/pablo/workspace/my-repo\n'); // toplevel succeeds
    
    expect(detectContext()).toBe('project:my-repo');
  });

  it('should fall back to cwd basename if git fails', () => {
    vi.mocked(execSync).mockImplementation(() => { throw new Error(); });
    const cwd = process.cwd();
    const expected = `project:${cwd.split('/').pop()}`;
    expect(detectContext()).toBe(expected);
  });

  it('should ensure context node exists', async () => {
    const contextId = await ensureContextNode('my-project');
    const nodes = await getAllNodes();
    expect(nodes).toHaveLength(1);
    expect(nodes[0].id).toBe(contextId);
    expect(nodes[0].label).toBe('[ctx:my-project]');
  });

  it('should bind node to context', async () => {
    const ctxId = await ensureContextNode('my-project');
    const node = await findOrCreateNode('node-1');
    bindToContext(node.id, ctxId);
  });
});
