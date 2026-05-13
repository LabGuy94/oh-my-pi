import { describe, expect, it } from "bun:test";
import { type MCPManager, SubAgentMCPProxy } from "../src/mcp/manager";
import { HttpTransport } from "../src/mcp/transports/http";
import type {
	MCPHttpServerConfig,
	MCPServerConfig,
	MCPServerConnection,
	MCPStdioServerConfig,
	MCPTransport,
} from "../src/mcp/types";

// SubAgentMCPProxy exists to give sub-agents their own MCP sessions (fresh
// `Mcp-Session-Id` for HTTP/SSE servers) instead of sharing the parent's
// transport instance. These tests pin that behavior with a thin in-memory
// stand-in for MCPManager — exercising the full MCPManager would require
// real network connections.

function stdioConnection(name = "stdio-server"): MCPServerConnection {
	return {
		name,
		config: { type: "stdio" as const, command: "echo" } satisfies MCPStdioServerConfig,
		transport: {
			connected: true,
			async request() {
				throw new Error("not used in test");
			},
			async notify() {},
			async close() {},
		},
		serverInfo: { name, version: "1.0" },
		capabilities: { tools: {} },
	};
}

function httpConnection(name = "http-server"): MCPServerConnection {
	const config: MCPHttpServerConfig = {
		type: "http" as const,
		url: "https://example.test/mcp",
	};
	// We need a real HttpTransport instance so SubAgentMCPProxy's
	// `instanceof HttpTransport` check fires. Don't call connect()/initialize();
	// the proxy never reaches into the transport, it only checks identity.
	const transport = new HttpTransport(config) as unknown as MCPTransport;
	return {
		name,
		config,
		transport,
		serverInfo: { name, version: "1.0" },
		capabilities: { tools: {} },
	};
}

/**
 * Stand-in for MCPManager that records calls to `connectIsolated` and lets
 * tests control what it returns. Only the methods SubAgentMCPProxy reaches
 * for are implemented; everything else throws.
 */
class FakeManager {
	readonly connections = new Map<string, MCPServerConnection>();
	readonly tools: ReturnType<MCPManager["getTools"]> = [];
	readonly isolatedCalls: Array<{ name: string; config: MCPServerConfig }> = [];
	connectIsolatedImpl: (name: string, config: MCPServerConfig) => Promise<MCPServerConnection> = async name => {
		throw new Error(`connectIsolated not stubbed for ${name}`);
	};

	getTools(): ReturnType<MCPManager["getTools"]> {
		return this.tools;
	}

	async waitForConnection(name: string): Promise<MCPServerConnection> {
		const conn = this.connections.get(name);
		if (!conn) throw new Error(`unknown server: ${name}`);
		return conn;
	}

	async connectIsolated(name: string, config: MCPServerConfig): Promise<MCPServerConnection> {
		this.isolatedCalls.push({ name, config });
		return this.connectIsolatedImpl(name, config);
	}

	asManager(): MCPManager {
		return this as unknown as MCPManager;
	}
}

describe("SubAgentMCPProxy", () => {
	it("returns the parent's connection for stdio servers (no session id concept)", async () => {
		const parent = new FakeManager();
		const stdio = stdioConnection();
		parent.connections.set(stdio.name, stdio);

		const proxy = new SubAgentMCPProxy(parent.asManager());

		const result = await proxy.waitForConnection(stdio.name);
		expect(result).toBe(stdio);
		expect(parent.isolatedCalls).toHaveLength(0);
	});

	it("opens an isolated connection for HTTP servers and caches it across calls", async () => {
		const parent = new FakeManager();
		const httpParent = httpConnection();
		parent.connections.set(httpParent.name, httpParent);

		const closed: string[] = [];
		const isolated = httpConnection();
		// Distinguish parent vs isolated by giving the isolated connection a
		// different transport identity.
		(isolated.transport as { close: () => Promise<void> }).close = async () => {
			closed.push("isolated");
		};
		parent.connectIsolatedImpl = async () => isolated;

		const proxy = new SubAgentMCPProxy(parent.asManager());
		const first = await proxy.waitForConnection(httpParent.name);
		const second = await proxy.waitForConnection(httpParent.name);

		expect(first).toBe(isolated);
		expect(second).toBe(isolated);
		expect(first).not.toBe(httpParent); // not the parent's connection
		expect(parent.isolatedCalls).toHaveLength(1);
		expect(parent.isolatedCalls[0]).toEqual({ name: httpParent.name, config: httpParent.config });
	});

	it("opens a distinct isolated connection per HTTP server", async () => {
		const parent = new FakeManager();
		const httpA = httpConnection("server-a");
		const httpB = httpConnection("server-b");
		parent.connections.set(httpA.name, httpA);
		parent.connections.set(httpB.name, httpB);

		const isoA = httpConnection("server-a-iso");
		const isoB = httpConnection("server-b-iso");
		parent.connectIsolatedImpl = async name => (name === "server-a" ? isoA : isoB);

		const proxy = new SubAgentMCPProxy(parent.asManager());
		const [a, b] = await Promise.all([proxy.waitForConnection("server-a"), proxy.waitForConnection("server-b")]);

		expect(a).toBe(isoA);
		expect(b).toBe(isoB);
		expect(parent.isolatedCalls.map(c => c.name).sort()).toEqual(["server-a", "server-b"]);
	});

	it("dedupes concurrent waits for the same server to a single connectIsolated call", async () => {
		const parent = new FakeManager();
		const httpParent = httpConnection();
		parent.connections.set(httpParent.name, httpParent);

		const isolated = httpConnection();
		const deferred = Promise.withResolvers<MCPServerConnection>();
		parent.connectIsolatedImpl = () => deferred.promise;

		const proxy = new SubAgentMCPProxy(parent.asManager());
		const first = proxy.waitForConnection(httpParent.name);
		const second = proxy.waitForConnection(httpParent.name);
		deferred.resolve(isolated);

		expect(await first).toBe(isolated);
		expect(await second).toBe(isolated);
		expect(parent.isolatedCalls).toHaveLength(1);
	});

	it("retries connectIsolated on the next call if the first attempt failed", async () => {
		const parent = new FakeManager();
		const httpParent = httpConnection();
		parent.connections.set(httpParent.name, httpParent);

		const isolated = httpConnection();
		let attempt = 0;
		parent.connectIsolatedImpl = async () => {
			attempt++;
			if (attempt === 1) throw new Error("connect refused");
			return isolated;
		};

		const proxy = new SubAgentMCPProxy(parent.asManager());
		await expect(proxy.waitForConnection(httpParent.name)).rejects.toThrow("connect refused");
		expect(await proxy.waitForConnection(httpParent.name)).toBe(isolated);
		expect(parent.isolatedCalls).toHaveLength(2);
	});

	it("dispose closes every opened isolated connection and is idempotent", async () => {
		const parent = new FakeManager();
		const httpA = httpConnection("server-a");
		const httpB = httpConnection("server-b");
		parent.connections.set(httpA.name, httpA);
		parent.connections.set(httpB.name, httpB);

		const closed: string[] = [];
		const isoA = httpConnection("server-a-iso");
		const isoB = httpConnection("server-b-iso");
		(isoA.transport as { close: () => Promise<void> }).close = async () => void closed.push("a");
		(isoB.transport as { close: () => Promise<void> }).close = async () => void closed.push("b");
		parent.connectIsolatedImpl = async name => (name === "server-a" ? isoA : isoB);

		const proxy = new SubAgentMCPProxy(parent.asManager());
		await Promise.all([proxy.waitForConnection("server-a"), proxy.waitForConnection("server-b")]);

		await proxy.dispose();
		expect(closed.sort()).toEqual(["a", "b"]);

		// Idempotent.
		await proxy.dispose();
		expect(closed.sort()).toEqual(["a", "b"]);
	});

	it("disposes connections that resolve after dispose was called", async () => {
		const parent = new FakeManager();
		const httpParent = httpConnection();
		parent.connections.set(httpParent.name, httpParent);

		const isolated = httpConnection();
		let closed = false;
		(isolated.transport as { close: () => Promise<void> }).close = async () => {
			closed = true;
		};
		const deferred = Promise.withResolvers<MCPServerConnection>();
		parent.connectIsolatedImpl = () => deferred.promise;

		const proxy = new SubAgentMCPProxy(parent.asManager());
		const pending = proxy.waitForConnection(httpParent.name);
		// Drain the parent.waitForConnection microtask so the proxy actually
		// reaches connectIsolated and our deferred.promise is the in-flight one.
		await Promise.resolve();

		// Start dispose without awaiting — dispose will block on the in-flight
		// connect, so we have to resolve it from the same turn.
		const disposed = proxy.dispose();
		deferred.resolve(isolated);
		await disposed;

		// Caller's promise rejects because we were disposed mid-connect.
		await expect(pending).rejects.toThrow(/disposed during connect/);
		// dispose awaited the connect, then ran transport.close on the result.
		expect(closed).toBe(true);
	});

	it("rejects waitForConnection after dispose", async () => {
		const parent = new FakeManager();
		const httpParent = httpConnection();
		parent.connections.set(httpParent.name, httpParent);

		const proxy = new SubAgentMCPProxy(parent.asManager());
		await proxy.dispose();

		await expect(proxy.waitForConnection(httpParent.name)).rejects.toThrow(/disposed/);
	});

	it("getTools delegates to the parent (definitions are shared; only connections differ)", () => {
		const parent = new FakeManager();
		// Cast through unknown — the test only checks identity, the tool shape
		// isn't exercised here.
		const tools = [{ name: "mcp__a__do" }, { name: "mcp__a__list" }] as unknown as ReturnType<MCPManager["getTools"]>;
		(parent.tools as unknown as typeof tools).push(...tools);

		const proxy = new SubAgentMCPProxy(parent.asManager());
		expect(proxy.getTools()).toBe(parent.tools);
	});
});
