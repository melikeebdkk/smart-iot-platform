using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using IoTPlatform.Api.Models;

namespace IoTPlatform.Api.Services
{
    public class AiService
    {
        private readonly HttpClient _httpClient;

        public AiService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        // -----------------------------
        // INSIGHTS
        // -----------------------------
        /// <summary>
        /// AI Service - Basic Power Insights
        /// </summary>
        public async Task<AiInsightsResponse?> GetBasicInsightsAsync(
            string deviceId,
            int hours = 24)
        {
            var url = $"/insights/basic?device_id={deviceId}&hours={hours}";

            var response = await _httpClient.GetAsync(url);
            if (!response.IsSuccessStatusCode)
                return null;

            return await response.Content
                .ReadFromJsonAsync<AiInsightsResponse>();
        }

        // -----------------------------
        // FORECAST
        // -----------------------------
        /// <summary>
        /// AI Service - Power Forecast
        /// </summary>
        public async Task<AiForecastResponse?> GetForecastAsync(
            string deviceId,
            int horizon = 5)
        {
            var url = $"/forecast?device_id={deviceId}&horizon={horizon}";

            var response = await _httpClient.GetAsync(url);
            if (!response.IsSuccessStatusCode)
                return null;

            return await response.Content
                .ReadFromJsonAsync<AiForecastResponse>();
        }

        // -----------------------------
        // ANOMALY
        // -----------------------------
        /// <summary>
        /// AI Service - Recent Anomalies
        /// </summary>
        public async Task<AiAnomalyResponse?> GetRecentAnomaliesAsync(
            string deviceId,
            int minutes = 30)
        {
            var url = $"/anomaly/recent?device_id={deviceId}&minutes={minutes}";

            var response = await _httpClient.GetAsync(url);
            if (!response.IsSuccessStatusCode)
                return null;

            return await response.Content
                .ReadFromJsonAsync<AiAnomalyResponse>();
        }
    }
}

