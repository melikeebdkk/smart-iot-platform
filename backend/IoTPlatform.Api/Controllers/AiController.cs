using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using IoTPlatform.Api.Services;

namespace IoTPlatform.Api.Controllers
{
    [ApiController]
    [Route("api/ai")]
    public class AiController : ControllerBase
    {
        private readonly AiService _aiService;

        public AiController(AiService aiService)
        {
            _aiService = aiService;
        }

        /// <summary>
        /// AI - Basic Power Insights
        /// </summary>
        [HttpGet("insights")]
        public async Task<IActionResult> GetInsights(
            [FromQuery] string deviceId,
            [FromQuery] int hours = 24)
        {
            var result = await _aiService.GetBasicInsightsAsync(deviceId, hours);

            if (result == null)
                return StatusCode(502, "AI service unavailable");

            return Ok(result);
        }

        /// <summary>
        /// AI - Forecast
        /// </summary>
        [HttpGet("forecast")]
        public async Task<IActionResult> GetForecast(
            [FromQuery] string deviceId,
            [FromQuery] int horizon = 5)
        {
            var result = await _aiService.GetForecastAsync(deviceId, horizon);

            if (result == null)
                return StatusCode(502, "AI service unavailable");

            return Ok(result);
        }

        /// <summary>
        /// AI - Recent Anomalies
        /// </summary>
        [HttpGet("anomaly")]
        public async Task<IActionResult> GetAnomalies(
            [FromQuery] string deviceId,
            [FromQuery] int minutes = 30)
        {
            var result = await _aiService.GetRecentAnomaliesAsync(deviceId, minutes);

            if (result == null)
                return StatusCode(502, "AI service unavailable");

            return Ok(result);
        }
    }
}

