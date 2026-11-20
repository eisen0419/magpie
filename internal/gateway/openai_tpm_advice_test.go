package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// yetone/magpie#7274: an OpenAI TPM 429 as a relay returned it, flat, not
// wrapped in "error". The limit is back in seconds. The last sentence is
// advice to add a payment method, and the link is account/billing.
const openAITPMPaymentAdvice = `{"message":"Rate limit reached for gpt-6.2-sol in organization <redacted> on tokens per min (TPM): Limit 1000000, Used 736214, Requested 401420. Please try again in 8.259s. Visit https://platform.openai.com/account/rate-limits to learn more. You can increase your rate limit by adding a payment method to your account at https://platform.openai.com/account/billing.","type":"tokens","param":null,"code":"rate_limit_exceeded"}`

// The same refusal in the envelope OpenAI's own API uses.
const openAITPMPaymentAdviceNested = `{"error":` + openAITPMPaymentAdvice + `}`

// A 429 that is money or a plan, not a minute's limit. Each stays
// unretried: the same-one retries and a patience's rounds both leave it.
func TestSpentMoneyIsNotAskedAgain(t *testing.T) {
	for _, tc := range []struct {
		name, body, why string
	}{
		{"insufficient quota", `{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}`, failCredit},
		{"billing hard limit", `{"error":{"message":"Billing hard limit has been reached.","type":"billing_limit_user_error","param":null,"code":"billing_hard_limit_reached"}}`, failQuota},
		{"balance depleted", `{"error":{"message":"Your account balance is depleted. Please top up.","code":"balance_depleted"}}`, failRate},
		{"quota and billing, no minute", `{"error":{"message":"You exceeded your current quota, please check your plan and billing details"}}`, failQuota},
		{"balance spent, back soon", `{"error":{"message":"Insufficient balance. Please try again in 30 seconds"}}`, failCredit},
		{"zhipu no balance", `{"error":{"code":"1113","message":"余额不足或无可用资源包,请充值。"}}`, failCredit},
		{"credits exhausted", wbExhaustedExplained, failCredit},
		{"per day, billing link", groqModelTPD, failQuota},
		{"token wall", raccoonWall, failRate},
		// a rate limit that only names credits, and is not per minute:
		// rested as a rate limit, and not asked again in place
		{"credits named in passing", `{"error":{"message":"Rate limit reached for requests; credits are not affected"}}`, failRate},
		// payment advice with no per-minute limit is the money, even
		// where the words also say rate limit
		{"payment advice, no minute", `{"error":{"message":"Rate limit exceeded. You can increase your rate limit by adding a payment method to your account."}}`, failRate},
		// a minute named beside a balance that is spent is the balance
		{"tpm and balance spent", `{"error":{"message":"Rate limit reached for gpt-6.2-sol on tokens per min (TPM): Limit 1000000. Insufficient balance. Please add a payment method.","code":"rate_limit_exceeded"}}`, failCredit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(tc.body)
			if got := failure(429, body); got != tc.why {
				t.Fatalf("failure: %s, want %s", got, tc.why)
			}
			if busy(429, body) {
				t.Error("busy: asked again in a patience's rounds")
			}
			if _, ok := passing(429, nil, body, 0); ok {
				t.Error("passes: asked again in place")
			}
		})
	}
}

// The reporter's TPM 429 is a rate limit, payment advice and all, so the
// same-one retries and a patience's rounds both take it. So is a minute's
// limit that says "billing" without the balance being spent (Gemini's
// PerMinute quota). A TPM limit with no such advice still is.
func TestOpenAITPMPaymentAdviceIsARateLimit(t *testing.T) {
	geminiPerMinute := ""
	for body := range minuteLimitBodies {
		if strings.Contains(body, "PerMinute") && strings.Contains(body, "billing") {
			geminiPerMinute = body
			break
		}
	}
	if geminiPerMinute == "" {
		t.Fatal("Gemini's per-minute fixture is gone")
	}
	for _, body := range []string{
		openAITPMPaymentAdvice,
		openAITPMPaymentAdviceNested,
		geminiPerMinute,
		openAIModelTPM,
	} {
		raw := []byte(body)
		if got := failure(429, raw); got != failRate {
			t.Errorf("failure: %s, want %s (%s)", got, failRate, body[:80])
		}
		if !busy(429, raw) {
			t.Errorf("not busy: %s", body[:80])
		}
		if _, ok := passing(429, nil, raw, 0); !ok {
			t.Errorf("doesn't pass: %s", body[:80])
		}
		if _, ok := passing(429, nil, raw, rateRetries); ok {
			t.Errorf("passes past the rate retries: %s", body[:80])
		}
	}
	// carried under a 200, the flat TPM limit is still that rate limit
	if got := failure(200, []byte(openAITPMPaymentAdvice)); got != failRate {
		t.Errorf("at 200: %s, want %s", got, failRate)
	}
}

// The rest is a rate limit's minute, not the half hour of a key out of credit.
func TestOpenAITPMPaymentAdviceRestsAMinute(t *testing.T) {
	old := allowances
	defer func() { allowances = old }()
	allowances = func(string) map[string]provider.Allowance { return nil }
	clineRestsCleared(t)
	s := &Server{}
	p := provider.Provider{ID: "openai", Chat: "https://api.openai.com/v1", Key: "sk-x"}
	c := candidate{p: p, model: "gpt-6.2-sol", rest: p.ID}
	r := s.restAfter(c, 429, nil, []byte(openAITPMPaymentAdvice))
	left := time.Until(r.Until)
	if r.Why != failRate || left > 2*time.Minute || left <= 0 {
		t.Fatalf("rests %v as %s by %s, want a rate limit's minute", left, r.Why, r.By)
	}
}

// One provider answers the reporter's 429, then the request. The agent
// gets the answer. The 429 is not the end of the turn.
func TestOpenAITPMPaymentAdviceAskedAgain(t *testing.T) {
	fresh(t)
	a := &scripted{replies: []reply{{429, "", openAITPMPaymentAdvice}, {200, "", chatOK}}}
	scriptedOn(t, "a", provider.Chat, a)
	s := New()
	code, body := postAs(t, s, "", `{"model":"a/m","messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 || !strings.Contains(body, "hello") || a.n != 2 {
		t.Fatalf("%d %s (asked %d times)", code, body, a.n)
	}
	try := firstTry(t, s)
	if try.Fail != failRate || try.Again <= 0 {
		t.Errorf("first try %+v: want a rate limit asked again", try)
	}
}

// A Retry-After past what the same-one retries wait for is held, and the
// provider is asked again once it has passed. The payment advice doesn't
// make the 429 one patience leaves.
func TestOpenAITPMPaymentAdviceWaitedOut(t *testing.T) {
	fresh(t)
	a := &scripted{
		replies: []reply{{429, "", openAITPMPaymentAdvice}, {200, "", chatOK}},
		limited: http.Header{"Retry-After": {"0.4"}},
	}
	scriptedOn(t, "a", provider.Chat, a)
	s := New()
	scaled(t, 3*time.Second)
	code, body := postAs(t, s, "", `{"model":"a/m","messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 || !strings.Contains(body, "hello") || a.n != 2 {
		t.Fatalf("%d %s (asked %d times)", code, body, a.n)
	}
	try := firstTry(t, s)
	if try.Fail != failRate || try.Again < 300 {
		t.Errorf("first try %+v: want it asked again after its Retry-After", try)
	}
}

// Money and a plan reach the agent after one ask. A 200 sitting behind
// the 429 is not fetched.
func TestSpentMoneyIsAskedOnce(t *testing.T) {
	for _, tc := range []struct {
		name, body string
	}{
		{"insufficient quota", `{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}`},
		{"billing hard limit", `{"error":{"message":"Billing hard limit has been reached.","type":"billing_limit_user_error","param":null,"code":"billing_hard_limit_reached"}}`},
		{"balance depleted", `{"error":{"message":"Your account balance is depleted. Please top up.","code":"balance_depleted"}}`},
		{"token wall", raccoonWall},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fresh(t)
			a := &scripted{replies: []reply{{429, "", tc.body}, {200, "", chatOK}}}
			scriptedOn(t, "a", provider.Chat, a)
			s := New()
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			type got struct {
				code int
				body string
			}
			done := make(chan got, 1)
			go func() {
				rec := httptest.NewRecorder()
				req := httptest.NewRequestWithContext(ctx, "POST", "/v1/chat/completions", strings.NewReader(`{"model":"a/m","messages":[{"role":"user","content":"hi"}]}`))
				s.Handler().ServeHTTP(rec, req)
				done <- got{rec.Code, rec.Body.String()}
			}()
			var g got
			deadline := time.Now().Add(2 * time.Second)
		wait:
			for {
				select {
				case g = <-done:
					break wait
				case <-time.After(5 * time.Millisecond):
				}
				if time.Now().After(deadline) {
					t.Fatalf("still open after %d asks", a.n)
				}
				s.trace.mu.Lock()
				waiting := false
				if k := len(s.trace.routes); k > 0 {
					for _, tr := range s.trace.routes[k-1].Tries {
						waiting = waiting || tr.Again > 0
					}
				}
				s.trace.mu.Unlock()
				if waiting {
					t.Fatalf("magpie waits to ask again: %+v", lastRoute(s).Tries)
				}
			}
			if g.code != 429 || a.n != 1 {
				t.Fatalf("agent got %d %s after %d asks; want the vendor's 429, asked once", g.code, g.body, a.n)
			}
		})
	}
}
