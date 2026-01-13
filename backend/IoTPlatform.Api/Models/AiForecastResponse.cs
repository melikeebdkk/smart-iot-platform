using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace IoTPlatform.Api.Models
{
    public class AiForecastResponse
    {
        [JsonPropertyName("device_id")]
        public string DeviceId { get; set; } = string.Empty;

        [JsonPropertyName("forecast")]
        public List<double> Forecast { get; set; } = new();
    }
}

