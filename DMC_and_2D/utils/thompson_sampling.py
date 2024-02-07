import random
import math


class ThompsonSampling:
	def __init__(self, m_a=0, v_a=10, alpha_a=1, beta_a=1):
		self.m_a = m_a
		self.v_a = v_a
		self.alpha_a = alpha_a
		self.beta_a = beta_a

		self.N = 0
		self.mean = 0
		self.rho = m_a
		self.ssd = 0
		self.beta_t_a = beta_a

		self.reset()

	def reset(self):
		self.N = 0
		self.mean = 0
		self.rho = self.m_a
		self.ssd = 0
		self.beta_t_a = self.beta_a

	def _draw_ig(self, alpha, beta):
		# draw from an inverse gamma with parameters alpha and beta
		try:
			return 1.0 / random.gammavariate(alpha, 1.0 / beta)
		except ZeroDivisionError:
			print("Failed for: " + self.label())
			raise

	def _draw_normal(self, mu, sigma2):
		# draw from a normal distribution with mean mu and *variance* sigma2
		return random.gauss(mu, math.sqrt(sigma2))

	def get_expected_reward(self):
		sigma2_a = self._draw_ig(0.5 * self.N + self.alpha_a, self.beta_t_a)
		mu_a = self._draw_normal(self.rho, sigma2_a / (self.N + self.v_a))
		return mu_a

	def record(self, reward):
		old_N, old_mean = self.N, self.mean
		self.N += 1
		self.mean += 1 / self.N * (reward - self.mean)
		self.rho = (self.v_a * self.m_a + self.N * self.mean) / (self.v_a + self.N)
		self.ssd += (reward ** 2 + old_N * old_mean ** 2 - self.N * self.mean ** 2)
		self.beta_t_a = (
			self.beta_a
			+ 0.5 * self.ssd
			+ (
				self.N
				* self.v_a
				* (self.mean - self.m_a) ** 2
				/ (2 * (self.N + self.v_a))
			)
		)

	def label(self):
		params = f"m = {self.m_a}, ν = {self.v_a}, α = {self.alpha_a}, β = {self.beta_a}"
		return "Thompson (%s)" % params

	def key(self):
		return f"TS_m{self.m_a}_nu{self.v_a}_alpha{self.alpha_a}_beta{self.beta_a}.csv"
