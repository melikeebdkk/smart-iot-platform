using System.Text.Json.Serialization;

namespace IoTPlatform.Api.Models
{
    public class AiInsightsResponse
    {
        [JsonPropertyName("device_id")]
        public string DeviceId { get; set; } = string.Empty;

        [JsonPropertyName("range_hours")]
        public int RangeHours { get; set; }

        [JsonPropertyName("avg_power")]
        public double AvgPower { get; set; }

        [JsonPropertyName("min_power")]
        public double MinPower { get; set; }

        [JsonPropertyName("max_power")]
        public double MaxPower { get; set; }

        [JsonPropertyName("total_consumption")]
        public double TotalConsumption { get; set; }

        [JsonPropertyName("data_points")]
        public int DataPoints { get; set; }
    }
}
